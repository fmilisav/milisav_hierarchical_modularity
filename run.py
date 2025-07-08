import os
import argparse
import json

from joblib import Parallel, delayed
from task import MemoryCapacity, EmpiricalMC, NLT, Multitasking, ChaoticPrediction, NeurogymTask
from network import SBM, Empirical
from sklearn.model_selection import ParameterSampler

import numpy as np
from scipy.stats import loguniform, uniform

import pickle

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

#Converts a string argument to a boolean value.
def str2bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("yes", "true", "t", "1"):
        return True
    elif arg.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

#Creates a directory if it does not exist.
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else: #add a number to the folder name if it already exists
        i = 1
        while os.path.exists(path + f'_{i}'):
            i += 1
        path = path + f'_{i}'
        os.makedirs(path)
    return path

#Builds the experiment name based on config settings
def create_exp_name(config):
    #Construct folder name based on experiment type and specific flags
    if config.experiment == 'SBM':
        return (f"{config.experiment}_{config.task}_{config.activation}"
                f"_nnodes{config.nnodes}_p1{config.p1}_delta_p{config.delta_p}"
                f"_min_weight{config.min_weight}"
                f"_bin_directed{config.bin_directed}"
                f"_wei_directed{config.wei_directed}")
    elif config.experiment == 'empirical':
        return f"{config.experiment}_{config.task}_gamma{config.gamma}"
    else:
        raise ValueError("Experiment not recognized. \
                         Choose between SBM and empirical.")

#Creates the directories for the experiment
def create_exp_dirs(config, exp_name):
    exp_path = os.path.join(config.main_dir, 'results', 
                            config.experiment, exp_name)
    exp_path = makedir(exp_path)
    config.net_path = os.path.join(exp_path, 'network_data')
    config.rs_path = os.path.join(exp_path, 'reservoir_states')
    config.LE_path = os.path.join(exp_path, 'LE')
    config.perform_path = os.path.join(exp_path, 'performance')
    os.makedirs(config.net_path)
    os.makedirs(config.rs_path)
    os.makedirs(config.LE_path)
    os.makedirs(config.perform_path)
    if config.task == 'NG':
        config.save_io_data_path = os.path.join(exp_path, 'io_data')
        os.makedirs(config.save_io_data_path)
    if (config.gain_optimization or
        config.ridge_optimization or
        config.density_optimization):
        config.opt_path = os.path.join(exp_path, 'hyperparameter_tuning')
        makedir(config.opt_path)

    return exp_path

#Run task analysis in parallel
def run_task_analysis(config, network_data, task):

    seeds = range(config.nnetworks)
    return Parallel(n_jobs=config.njobs, verbose=1)(
            delayed(task.analysis)(*network_data, seed) for seed in seeds)
    
def init_param_sampler(niter, sampler_seed,
                       gain_extrema=None, 
                       ridge_extrema=None,
                       pruning_extrema=None):

    param_distribs = {}
    if gain_extrema is not None:
        param_distribs['input_gain'] = loguniform(*gain_extrema)
    if ridge_extrema is not None:
        param_distribs['l2_alpha'] = loguniform(*ridge_extrema)
    if pruning_extrema is not None:
        param_distribs['pruning_ratio'] = uniform(*pruning_extrema)

    param_sampler = ParameterSampler(param_distribs, n_iter=niter, 
                                     random_state=sampler_seed)
    return list(param_sampler)

def save_hyperparameter_results(results):

    with open(os.path.join(config.opt_path,
              'hyperparameter_results.npy'), 'wb') as f:
        pickle.dump(results, f)

#Run hyperparameter optimization
def hyperparameter_opt(config, task, net_params, opt_params,
                       gain_extrema=None, ridge_extrema=None,
                       pruning_extrema=None):
    
    niter, nseeds, sampler_seed = opt_params
    
    #Sample hyperparameters
    param_sampler = init_param_sampler(niter, sampler_seed,
                                       gain_extrema=gain_extrema,
                                       ridge_extrema=ridge_extrema,
                                       pruning_extrema=pruning_extrema)
    if gain_extrema is None:
        for i in range(niter):
            param_sampler[i]['input_gain'] = config.input_gain
    if ridge_extrema is None:
        for i in range(niter):
            param_sampler[i]['l2_alpha'] = config.l2_alpha
    if pruning_extrema is None:
        for i in range(niter):
            param_sampler[i]['pruning_ratio'] = config.pruning_ratio
    task.param_sampler = param_sampler
    
    #Initialize neurogym data if needed
    if config.task == 'NG':
        for seed in range(nseeds):
            task.seed = seed + 2*len(net_params[0]['1'])
            task.init_data(nseeds)

    #Run hyperparameter optimization in parallel
    scores_niter = Parallel(n_jobs=config.njobs, verbose=1)(
                   delayed(task.hyperparameter_opt)(*net_params, nseeds,
                           input_gain=param_sampler[i]['input_gain'],
                           l2_alpha=param_sampler[i]['l2_alpha'],
                           pruning_ratio=param_sampler[i]['pruning_ratio']) for i in range(niter))
        
    task.hyperparameter_results = scores_niter
    save_hyperparameter_results(scores_niter)

    #Select the best hyperparameters
    input_gain, l2_alpha, pruning_ratio = task.best_hyperparameter(config.niter, config.n_opt_seeds,
                                                                   aggregate=config.in_out_agg,
                                                                   gain_opt=config.gain_optimization,
                                                                   ridge_opt=config.ridge_optimization,
                                                                   density_opt=config.density_optimization)
    if config.gain_optimization:
        config.input_gain = input_gain
    if config.ridge_optimization:
        config.l2_alpha = l2_alpha
    if config.density_optimization:
        config.pruning_ratio = pruning_ratio

#Run the experiment.
def run_exp(config):

    config.main_dir = os.getcwd()
    exp_name = create_exp_name(config)

    exp_path = create_exp_dirs(config, exp_name)

    gain_extrema = None
    ridge_extrema = None
    pruning_extrema = None
    if config.gain_optimization:
        gain_extrema = [config.gain_min, config.gain_max]
    if config.ridge_optimization:
        ridge_extrema = [config.ridge_min, config.ridge_max]
    if config.density_optimization:
        pruning_extrema = [config.pruning_min, config.pruning_max]
    if (config.gain_optimization or config.ridge_optimization or 
        config.density_optimization):
        opt_params = [config.niter, config.n_opt_seeds, config.sampler_seed]

    with open(os.path.join(exp_path, 'config.txt'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    if config.experiment == 'SBM':

        task_mapping = {'MC': MemoryCapacity, 'NLT': NLT, 'MT': Multitasking, 
                        'CP': ChaoticPrediction, 'NG': NeurogymTask}
        if config.task not in task_mapping:
            raise ValueError("Task not recognized. \
                             Choose MC, NLT, MT, CP, or NG.")

        #Initialize the task
        if (config.task == 'CP' or config.task == 'NG'):
            task = task_mapping[config.task](config, **config.kwargs)
        else:
            task = task_mapping[config.task](config)

        #Initialize the SBM
        sbm = SBM(config)
        sbm_params = [sbm.nets, sbm.nnodes, sbm.nodes, 
                      sbm.nmodules, sbm.module_mappings]

        if config.save_nets:
            sbm.save_data(sbm.nets, 'nets.pickle')
        
        if (config.ridge_optimization or config.gain_optimization 
            or config.density_optimization):
            #Hyperparameter tuning
            hyperparameter_opt(config, task, sbm_params, opt_params,
                               gain_extrema=gain_extrema,
                               ridge_extrema=ridge_extrema,
                               pruning_extrema=pruning_extrema)

        with open(os.path.join(exp_path, 'config.txt'), 'w') as f:
            json.dump(config.__dict__, f, indent=2)
            f.flush()  # Force writing to disk
            os.fsync(f.fileno())

        run_task_analysis(config, sbm_params, task)

    elif config.experiment == 'empirical':

        w = np.load(config.data_path)
        #Initialize the empirical data object
        empirical = Empirical(config, w, net_data_path=config.net_data_path)
        #Save the empirical data if it does not exist
        if config.net_data_path is None:
            empirical.save_data(empirical.zrands, 'zrands.pickle')
            empirical.save_data(empirical.max_zrand_idx, 'max_zrand_idx.pickle')
            empirical.save_data(empirical.module_mappings, 'module_mappings.pickle')
            empirical.save_data(empirical.module_mappings_level2, 'module_mappings_level2.pickle')
            empirical.save_data(empirical.strength, 'strength.pickle')
            empirical.save_data(empirical.mod_strength, 'mod_strength.pickle')
            empirical.save_data(empirical.clustering, 'clustering.pickle')
            empirical.save_data(empirical.mod_clustering, 'mod_clustering.pickle')
            empirical.save_data(empirical.communicability, 'communicability.pickle')
            empirical.save_data(empirical.mod_communicability, 'mod_communicability.pickle')
        if config.save_nets:
            empirical.save_data(empirical.nets, 'nets.pickle')

        #Choosing the number of output nodes per module as
        #the minimum of 10 or the minimum number of nodes in a module
        modules, counts = np.unique(empirical.module_mappings, 
                                    return_counts=True)
        noutputs = np.min([10, np.min(counts)])
        empirical_params = [empirical.nets, empirical.module_mappings, 
                            noutputs]

        if config.task == 'MC':

            task = EmpiricalMC(config)

            #Hyperparameter tuning
            if config.ridge_optimization or config.gain_optimization:
                hyperparameter_opt(config, task, empirical_params, opt_params,
                                   gain_extrema=gain_extrema, 
                                   ridge_extrema=ridge_extrema)

            with open(os.path.join(exp_path, 'config.txt'), 'w') as f:
                json.dump(config.__dict__, f, indent=2)

            run_task_analysis(config, empirical_params, task)

        else: raise ValueError('Empirical data only supports MC task.')

    with open(os.path.join(exp_path, 'config.txt'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RC experiment')

    parser.add_argument('--experiment', type=str, default='SBM', help='Experiment to run')
    parser.add_argument('--njobs', type=int, default=50, help='Number of parallel jobs')

    parser.add_argument('--nnetworks', type=int, default=100, help='Number of networks to generate')
    parser.add_argument('--duration', type=int, default=2000, help='Time duration of the simulation')
    parser.add_argument('--warmup', type=int, default=100, help='Warmup time to discard')
    parser.add_argument('--l2_alpha', type=float, default=1.0, help='L2 regularization parameter')
    parser.add_argument('--gain_optimization', type=str2bool, default=False, help='Whether to optimize the input gain parameter or not')
    parser.add_argument('--ridge_optimization', type=str2bool, default=False, help='Whether to optimize the ridge parameter or not')
    parser.add_argument('--density_optimization', type=str2bool, default=False, help='Whether to optimize the pruning ratio or not')
    parser.add_argument('--alphas', type=list_of_floats, default=[0.6, 0.8, 1.0, 1.2, 1.4], help='List of alpha parameters')
    parser.add_argument('--input_gain', type=float, default=1.0, help='Input gain')
    parser.add_argument('--pruning_ratio', type=float, default=0.0, help='Pruning ratio')
    parser.add_argument('--nedges', type=int, default=None, help='Expected number of edges')
    parser.add_argument('--compute_LE', type=str2bool, default=True, help='Whether to compute Lyapunov Exponents or not')
    parser.add_argument('--alphas_to_save', type=list_of_floats, default=[0.8, 1.0, 1.2], help='List of alpha parameters for which to save timeseries')
    parser.add_argument('--criticality', type=float, default=1, help='Critical alpha')
    parser.add_argument('--activation', type=str, default='tanh', help='Activation function')
    parser.add_argument('--score', type=str, default='score', help='Performance metric')
    parser.add_argument('--multioutput', type=str, default='uniform_average', help='Multioutput strategy')

    task_subparsers = parser.add_subparsers(dest='task', help='Task to run')
    mc_parser = task_subparsers.add_parser('MC', help='Memory Capacity')
    mc_parser.add_argument('--horizon_max', type=int, default=-16, help='Maximum time-lag to evaluate MC')
    mc_parser.add_argument('--input_amp', type=str2bool, default=True, help='Whether to use input amplification or not')
    mc_parser.add_argument('--training_noise', type=str2bool, default=False, help='Whether to add noise to the training data or not')
    mc_parser.add_argument('--testing_noise', type=str2bool, default=False, help='Whether to add noise to the testing data or not')
    mc_parser.add_argument('--max_noise', type=float, default=0.1, help='Maximum noise amplitude')

    nlt_parser = task_subparsers.add_parser('NLT', help='Nonlinear Transformation')
    nlt_parser.add_argument('--ncycles', type=int, default=10, help='Number of cycles in sinusoidal signal for NLT task')
    nlt_parser.add_argument('--lag', type=str2bool, default=False, help='Whether to use time-lag')
    nlt_parser.add_argument('--input_amp', type=str2bool, default=True, help='Whether to use input amplification or not')

    multitasking_parser = task_subparsers.add_parser('MT', help='Multitasking')
    multitasking_parser.add_argument('--horizon_max', type=int, default=-16, help='Maximum time-lag to evaluate MC')
    multitasking_parser.add_argument('--ncycles', type=list_of_ints, default=[10, 20, 30, 40], help='Number of cycles in sinusoidal signal for NLT task')
    multitasking_parser.add_argument('--lag', type=str2bool, default=False, help='Whether to use time-lag')
    multitasking_parser.add_argument('--ninputs', type=int, default=8, help='Number of inputs')
    multitasking_parser.add_argument('--interleaved', type=str2bool, default=False, help='Whether to interleave the tasks or not')
    multitasking_parser.add_argument('--training_noise', type=str2bool, default=False, help='Whether to add noise to the training data or not')
    multitasking_parser.add_argument('--testing_noise', type=str2bool, default=False, help='Whether to add noise to the testing data or not')
    multitasking_parser.add_argument('--max_noise', type=float, default=0.1, help='Maximum noise amplitude')

    cp_parser = task_subparsers.add_parser('CP', help='Chaotic Prediction')
    cp_parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon')
    cp_parser.add_argument('--task_name', type=str, default='narma', help='Task name')
    cp_parser.add_argument('--init_min', type=int, default=0, help='Minimum initial condition')
    cp_parser.add_argument('--init_max', type=int, default=1, help='Maximum initial condition')
    cp_parser.add_argument('--input_amp', type=str2bool, default=False, help='Whether to use input amplification or not')
    cp_parser.add_argument('--distributed_input', type=str2bool, default=False, help='Whether to use a distributed input or not')
    cp_parser.add_argument('--training_noise', type=str2bool, default=False, help='Whether to add noise to the training data or not')
    cp_parser.add_argument('--noise_factor', type=float, default=0.1, help='Noise factor')
    cp_parser.add_argument('--test_split', type=int, default=None, help='Test split duration')
    cp_parser.add_argument('--kwargs', type=json.loads, default='{}', help='Additional key=value arguments for CP task')

    ng_parser = task_subparsers.add_parser('NG', help='NeuroGym')
    ng_parser.add_argument('--load_io_data_path', type=str, default=None, help='Path to load input-output data')
    ng_parser.add_argument('--task_name', type=str, default='HierarchicalReasoning', help='Task name')
    ng_parser.add_argument('--input_amp', type=str2bool, default=False, help='Whether to use input amplification or not')
    ng_parser.add_argument('--distributed_input', type=str2bool, default=False, help='Whether to use a distributed input or not')
    ng_parser.add_argument('--sample_weight_strat', type=str, default='decision', help='Sample weight strategy: "whole" weights fixation randomly, \
                                                                                       "decision" weights samples by decision time')
    ng_parser.add_argument('--grace_period', type=int, default=4, help='Grace period before evaluating')
    ng_parser.add_argument('--training_noise', type=str2bool, default=False, help='Whether to add noise to the training data or not')
    ng_parser.add_argument('--testing_noise', type=str2bool, default=False, help='Whether to add noise to the testing data or not')
    ng_parser.add_argument('--max_noise', type=float, default=0.1, help='Maximum noise amplitude')
    ng_parser.add_argument('--kwargs', type=json.loads, default='{}', help='Additional key=value arguments for NG task')

    config, remaining = parser.parse_known_args()

    if config.experiment == 'SBM':
        parser.add_argument('--nnodes', type=int, default=50, help='Number of nodes in each block')
        parser.add_argument('--p1', type=float, default=0.5, help='Probability of connection within-block')
        parser.add_argument('--delta_p', type=float, default=0.5, help='Factor of decrease for connection probabilities')
        parser.add_argument('--min_weight', type=float, default=0, help='Minimum weight of connections')
        parser.add_argument('--bin_directed', type=str2bool, default=False, help='Whether the networks are directed or not')
        parser.add_argument('--wei_directed', type=str2bool, default=False, help='Whether the weights are directed/asymmetric or not')
        parser.add_argument('--selfloops', type=str2bool, default=False, help='Whether to include self-loops or not')
        parser.add_argument('--i_ratio', type=float, default=0, help='Ratio of inhibitory connections')
        parser.add_argument('--save_nets', type=str2bool, default=False, help='Whether to save the networks or not')
    elif config.experiment == 'empirical':
        parser.add_argument('--data_path', type=str, help='Path to the empirical data', default='data/SC_wei_HCP_s400.npy')
        parser.add_argument('--net_data_path', type=str, help='Path to the empirical network data', default=None)
        parser.add_argument('--gamma', type=float, default=1, help='Resolution parameter for modularity maximization')
        parser.add_argument('--nseeds', type=int, default=1000, help='Number of realizations of modularity maximization')
        parser.add_argument('--save_nets', type=str2bool, default=False, help='Whether to save the networks or not')
    else: raise ValueError('Experiment not recognized. Please choose between SBM and empirical.')

    if config.gain_optimization:
        parser.add_argument('--gain_min', type=float, default=1e-10, help='Minimum value of the input gain parameter')
        parser.add_argument('--gain_max', type=float, default=10, help='Maximum value of the input gain parameter')
    if config.ridge_optimization:
        parser.add_argument('--ridge_min', type=float, default=1e-10, help='Minimum value of the ridge parameter')
        parser.add_argument('--ridge_max', type=float, default=10, help='Maximum value of the ridge parameter')
    if config.density_optimization:
        parser.add_argument('--pruning_min', type=float, default=0, help='Minimum value of the pruning ratio parameter')
        parser.add_argument('--pruning_max', type=float, default=1, help='Maximum value of the pruning ratio parameter')
    if config.gain_optimization or config.ridge_optimization or config.density_optimization:
        parser.add_argument('--niter', type=int, default=125, help='Number of iterations for the hyperparameter optimization')
        parser.add_argument('--n_opt_seeds', type=int, default=25, help='Number of seeds for the hyperparameter optimization')
        parser.add_argument('--sampler_seed', type=int, default=0, help='Seed for the sampler')
        parser.add_argument('--in_out_agg', type=str, default='average', help='Aggregation method across input-output pairs')

    #check that alphas_to_save is subset of alphas
    if not set(config.alphas_to_save).issubset(set(config.alphas)):
        raise ValueError('alphas_to_save must be a subset of alphas')
    #check that alphas_to_save contains criticality, subcriticality and supercriticality, if compute_LE is True
    if config.compute_LE:
        if config.criticality not in config.alphas_to_save:
            raise ValueError('alphas_to_save must contain criticality')
        if not(np.any(np.array(config.alphas_to_save) < config.criticality)):
            raise ValueError('alphas_to_save must contain subcriticality')
        if not(np.any(np.array(config.alphas_to_save) > config.criticality)):
            raise ValueError('alphas_to_save must contain supercriticality')

    #check that MC is used for empirical experiment
    if config.experiment == 'empirical' and config.task != 'MC':
        raise ValueError('Empirical data only supports MC task.')
    #check that density is not optimized for empirical experiment
    if config.experiment == 'empirical' and config.density_optimization:
        raise ValueError('Density optimization is not supported for empirical data.')
    
    #warn that if nedges is passed, p1 will be overwritten based on
    #the expected number of edges, the number of nodes, and the delta_p
    if config.nedges is not None:
        print('Warning: p1 will be overwritten based on the expected number of edges, the number of nodes, and the delta_p.')

    #if multitasking, check that ninputs is 2, 4, or 8
    if config.task == 'MT':
        if config.ninputs not in [2, 4, 8]:
            raise ValueError('ninputs must be 2, 4, or 8 for multitasking task.')

    parser.parse_args(remaining, namespace=config)

    run_exp(config)