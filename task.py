from abc import ABC

import numpy as np
from scipy import signal

import pandas as pd

import bct
import networkx as nx

from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout, _get_sample_weight
from conn2res.tasks import Conn2ResTask, ReservoirPyTask, NeuroGymTask

from sklearn.linear_model import Ridge, RidgeClassifier

import os
import pickle

class Task(ABC):

    """
    Abstract base class for running tasks and hyperparameter optimization

    Attributes
    ----------
    duration : int
        Number of time steps on which the readout is trained
    warmup : int
        Number of initial time steps to discard before training the readout
    total_duration : int
        Total number of time steps in the task
    alphas : list
        List of reservoir spectral radii to test
    l2_alpha : float
        Ridge regularization parameter
    input_gain : float
        Input gain for the reservoir
    pruning_ratio : float
        Ratio of connections to prune
    compute_LE : bool
        Whether to compute the Lyapunov exponents
    alphas_to_save : list
        List of spectral radii for which to save the reservoir states
    criticality : float
        Critical spectral radius
    activation : str
        Activation function for the reservoir
    score : str
        Scoring method for the readout
    multioutput : str
        Multioutput strategy for the readout
    rs_path : str
        Path to save reservoir states
    LE_path : str
        Path to save Lyapunov exponents
    perform_path : str
        Path to save performance results

    Methods
    -------
    init_conn(seed, nets, level)
        Initialize a connectivity matrix
    init_net_id(level, module=None)
        Initialize a network ID
    simulation(alpha, conn, w_in, output_nodes, compute_LE, 
               sample_weight, multioutput, task, readout_modules=None)
        Run a simulation
    task_workflow(conn, w_in, output_nodes, readout_modules, net_id=None,
                  compute_LE=False, sample_weight=None, 
                  multioutput='uniform_average', task='MC')
        Run a task
    alpha_to_regime(alpha)
        Return the dynamical regime of the reservoir
    save_rs(esn, alpha, net_id)
        Save reservoir states
    save_LEs(LEs, LEs_trajectory, alpha, net_id)
        Save Lyapunov exponents
    save_results()
        Save performance results
    best_hyperparameter(niter, nseeds, aggregate='average',
                        gain_opt=False, ridge_opt=False,
                        density_opt=False)
        Return the best hyperparameters
    """

    def __init__(self, config):

        self.duration = config.duration
        self.warmup = config.warmup
        self.total_duration = self.duration + self.warmup
        self.alphas = config.alphas
        self.l2_alpha = config.l2_alpha
        self.input_gain = config.input_gain
        self.pruning_ratio = config.pruning_ratio
        self.compute_LE = config.compute_LE
        self.alphas_to_save = config.alphas_to_save
        self.criticality = config.criticality
        self.activation = config.activation
        self.score = config.score
        self.multioutput = config.multioutput
        self.rs_path = config.rs_path
        self.LE_path = config.LE_path
        self.perform_path = config.perform_path

    def init_conn(self, seed, nets, level):

        w = np.array(nets[level][seed])
        #delete self.pruning_ratio of connections
        if self.pruning_ratio > 0:
            #directed
            if not np.allclose(w, w.T):
                idx = np.where(w != 0)
                nedges = len(idx[0])
                nprune = int(self.pruning_ratio*nedges)
                idx_prune = np.random.choice(range(nedges),
                                             nprune, replace=False)
                w[idx[0][idx_prune], idx[1][idx_prune]] = 0

                #check connectedness
                if not nx.is_strongly_connected(nx.DiGraph(w)):
                    return None
            #undirected
            else:
                #upper triangle
                triu_w = np.triu(w)
                idx = np.where(triu_w != 0)
                nedges = len(idx[0])
                nprune = int(self.pruning_ratio*nedges)
                idx_prune = np.random.choice(range(nedges),
                                             nprune, replace=False)
                triu_w[idx[0][idx_prune], idx[1][idx_prune]] = 0
                #add back the lower triangle
                w = triu_w + np.tril(triu_w.T, -1)

                #check connectedness
                if bct.number_of_components(w) > 1:
                    return None

        conn = Conn(w=w)
        #normalize the connectivity matrix to have a spectral radius of 1
        conn.normalize()

        return conn

    def init_net_id(self, level, module=None):

        if module is not None:
            net_id = '_level{}_module{}_{}.npy'.format(level, module,
                                                       self.seed)
        else:
            net_id = '_level{}_{}.npy'.format(level, self.seed)

        return net_id

    def simulation(self, alpha, conn, w_in,
                   output_nodes, compute_LE, 
                   multioutput, task, 
                   readout_modules=None,
                   sample_weight=None):

        esn = EchoStateNetwork(w=alpha*conn.w,
                               activation_function=self.activation)

        rs_train = esn.simulate(
            ext_input=self.x_train, w_in=w_in, input_gain=self.input_gain,
            output_nodes=output_nodes, compute_LE=False, warmup=self.warmup
        )

        rs_train = rs_train[self.warmup:]

        rs_test = esn.simulate(
            ext_input=self.x_test, w_in=w_in, input_gain=self.input_gain,
            output_nodes=output_nodes, compute_LE=compute_LE, 
            warmup=self.warmup
        )

        rs_test = rs_test[self.warmup:]

        if task != 'MT':
            df_res = self.readout.run_task(
                X=(rs_train, rs_test), y=(self.y_train, self.y_test),
                sample_weight=sample_weight, metric=self.score, 
                readout_modules=readout_modules, multioutput=multioutput
            )

        if task == 'MT':
            return esn, rs_train, rs_test
        else:
            return esn, df_res

    def task_workflow(self, conn, w_in,
                      output_nodes, readout_modules,
                      net_id=None, compute_LE=False,
                      sample_weight=None, 
                      multioutput='uniform_average',
                      task='single'):

        df_alpha = []
        for alpha in self.alphas:

            esn, df_res = self.simulation(alpha, conn, w_in,
                                          output_nodes, compute_LE, 
                                          multioutput, task, 
                                          readout_modules=readout_modules,
                                          sample_weight=sample_weight)
            
            if net_id is not None:
                self.save_rs(esn, alpha, net_id)

            if compute_LE:
                self.save_LEs(esn, alpha, net_id)

            df_res['alpha'] = alpha
            df_alpha.append(df_res)
        df_alpha = pd.concat(df_alpha, ignore_index=True)

        return df_alpha

    def alpha_to_regime(self, alpha):

        if alpha < self.criticality:
            return 'stable'
        elif alpha == self.criticality:
            return 'critical'
        else:
            return 'chaotic'

    def save_rs(self, esn, alpha, net_id):

        if self.alphas_to_save is not None and self.rs_path is not None:
            if alpha in self.alphas_to_save:
                regime = self.alpha_to_regime(alpha)
                with open(os.path.join(self.rs_path, regime +
                                        '_rs_alpha{}'.format(alpha) +
                                        net_id), 'wb') as f:
                    np.save(f, esn._state[self.warmup:])
    #warmup kept in SBM_MC_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse_3

    def save_LEs(self, esn, alpha, net_id):

        if self.compute_LE and self.LE_path is not None:
            with open(os.path.join(self.LE_path, 'LEs_alpha{}'.format(alpha) +
                                   net_id), 'wb') as f:
                np.save(f, esn.LE)
            with open(os.path.join(self.LE_path, 'LEs_trajectory_alpha{}'.format(alpha) +
                                   net_id), 'wb') as f:
                np.save(f, esn.LE_trajectory)

    def save_results(self):

        with open(os.path.join(self.perform_path,
                               'results_seed{}.npy'.format(self.seed)), 'wb') as f:
            pickle.dump(self.results, f)

    #scores are averaged or maxed across output modules
    #maxed across alpha values
    #averaged across seeds
    #maxed across network types
    #finally, the highest performing parameter is chosen
    def best_hyperparameter(self, niter, nseeds,
                            aggregate='average',
                            gain_opt=False, ridge_opt=False,
                            density_opt=False):

        input_gain = None
        l2_alpha = None
        pruning_ratio = None

        scores = []
        for scores_niter in self.hyperparameter_results:
            if scores_niter is None:
                scores.append(np.nan)
                continue
            scores_type = []
            for net_type in range(4):
                max_scores = []
                for seed in range(nseeds):
                    #aggregate scores across output modules
                    if aggregate == 'average':
                        df_agg = scores_niter[seed][net_type].groupby('alpha').agg({self.score: 'mean'}).reset_index()
                    elif aggregate == 'max':
                        df_agg = scores_niter[seed][net_type].groupby('alpha').agg({self.score: 'max'}).reset_index()
                    else:
                        raise ValueError("Invalid aggregation method. "\
                                            "Choose from 'average' or 'max'.")
                    #max score across alpha values
                    max_scores.append(df_agg[self.score].max())
                #mean score across seeds
                scores_type.append(np.mean(max_scores))
            #max score across network types
            scores.append(np.max(scores_type))
        #best hyperparameter
        if gain_opt:
            self.input_gain = self.param_sampler[np.nanargmax(scores)]['input_gain']
            input_gain = self.input_gain
        if ridge_opt:
            self.l2_alpha = self.param_sampler[np.nanargmax(scores)]['l2_alpha']
            l2_alpha = self.l2_alpha
        if density_opt:
            self.pruning_ratio = self.param_sampler[np.nanargmax(scores)]['pruning_ratio']
            pruning_ratio = self.pruning_ratio

        return input_gain, l2_alpha, pruning_ratio

class RegressionUtils(ABC):

    """
    Intermediate abstract class for
    Regression tasks

    Attributes
    ----------
    seed : int
        Random seed for the task
    readout : Readout
        Readout object for the task
    results : dict
        Performance results for the task
    input_gain : float
        Input gain for the reservoir
    l2_alpha : float
        Ridge regularization parameter
    pruning_ratio : float
        Ratio of connections to prune

    Methods
    -------
    set_in_out(seed, nnodes, nodes, nmodules, module_mappings, conn)
        Set input and output nodes
    analysis(sbms, nnodes, nodes, nmodules, module_mappings, seed)
        Run the task
    hyperparameter_opt(sbms, nnodes, nodes, nmodules, module_mappings, 
                       nseeds, input_gain=None, l2_alpha=None,
                       pruning_ratio=None)
        Run hyperparameter optimization
    """

    def set_in_out(self, seed, nnodes, nodes,
                   nmodules, module_mappings, conn):

        np.random.seed(seed)
        module = np.random.randint(nmodules)

        input_nodes = np.array(range(module*nnodes, module*nnodes + nnodes))
        if not self.input_amp:
            input_nodes = conn.get_nodes(
                        'random', nodes_from=input_nodes, seed=seed
                    )
        
        output_nodes = (nodes < module*nnodes)|(nodes >= module*nnodes+nnodes)
        readout_modules = module_mappings[output_nodes]

        w_in = np.zeros((1, conn.n_nodes))
        w_in[:, input_nodes] = 1

        return module, output_nodes, readout_modules, w_in
    
    def analysis(self, sbms, nnodes, nodes,
                 nmodules, module_mappings, seed):

        print('Running seed {}'.format(seed))
        self.seed = seed

        self.init_data(len(sbms['1']))

        self.readout = Readout(estimator=Ridge(alpha=self.l2_alpha,
                                               fit_intercept=False))
        
        scores = {}
        for level in sbms.keys():

            conn = self.init_conn(self.seed, sbms, level)
            module, output_nodes, readout_modules, w_in = self.set_in_out(self.seed, nnodes, nodes,
                                                                          nmodules, module_mappings,
                                                                          conn)
            net_id = self.init_net_id(level, module=module)
            df_alpha = self.task_workflow(conn, w_in, output_nodes,
                                          readout_modules, net_id=net_id,
                                          compute_LE=self.compute_LE,
                                          multioutput=self.multioutput)
            scores[level] = df_alpha
        self.results = scores

        self.save_results()

    def hyperparameter_opt(self, sbms, nnodes, nodes,
                           nmodules, module_mappings, nseeds,
                           input_gain=None,
                           l2_alpha=None,
                           pruning_ratio=None):
        
        if input_gain is not None:
            self.input_gain = input_gain
        if l2_alpha is not None:
            self.l2_alpha = l2_alpha
        if pruning_ratio is not None:
            self.pruning_ratio = pruning_ratio

        self.readout = Readout(estimator=Ridge(alpha=self.l2_alpha,
                                               fit_intercept=False))
        
        scores_seed = []
        for seed in range(nseeds):

            self.seed = seed + 2*len(sbms['1'])
            self.init_data(nseeds)

            scores_net_type = []
            for net_type in sbms.keys():

                conn = self.init_conn(seed, sbms, net_type)
                if conn is None:
                    return None

                module, output_nodes, readout_modules, w_in = self.set_in_out(seed, nnodes, nodes,
                                                                              nmodules, module_mappings,
                                                                              conn)
                df_alpha = self.task_workflow(conn, w_in, output_nodes, 
                                              readout_modules)
                scores_net_type.append(df_alpha)
            scores_seed.append(scores_net_type)

        return scores_seed

class ClassifierUtils(ABC):

    """
    Intermediate abstract class for
    Classification tasks

    Attributes
    ----------
    seed : int
        Random seed for the task
    readout : Readout
        Readout object for the task
    results : dict
        Performance results for the task
    input_gain : float
        Input gain for the reservoir
    l2_alpha : float
        Ridge regularization parameter
    pruning_ratio : float
        Ratio of connections to prune

    Methods
    -------
    sample_weight()
        Get sample weights for the task
    analysis(sbms, nnodes, nodes, nmodules, module_mappings, seed)
        Run the task
    hyperparameter_opt(sbms, nnodes, nodes, nmodules, module_mappings,
                       nseeds, input_gain=None, l2_alpha=None,
                       pruning_ratio=None)
        Run hyperparameter optimization
    """

    def sample_weight(self):

        if self.sample_weight_strat == 'whole':
            sample_weight_train = _get_sample_weight(self.y_train, split_set='train', 
                                                     grace_period=self.grace_period,
                                                     seed=self.seed)
            sample_weight_test = _get_sample_weight(self.y_test, split_set='test',
                                                    grace_period=self.grace_period)
        else:
            sample_weight_train, sample_weight_test = _get_sample_weight((self.y_train, self.y_test),
                                                                         grace_period=self.grace_period)
        sample_weight = (sample_weight_train, sample_weight_test)

        return sample_weight
    
    def analysis(self, sbms, nnodes, nodes,
                 nmodules, module_mappings, seed):

        print('Running seed {}'.format(seed))
        self.seed = seed

        self.init_data(len(sbms['1']))
        sample_weight = self.sample_weight()

        self.readout = Readout(estimator=RidgeClassifier(alpha=self.l2_alpha,
                                                         fit_intercept=False))
        
        scores = {}
        for level in sbms.keys():

            conn = self.init_conn(self.seed, sbms, level)
            module, output_nodes, readout_modules, w_in = self.set_in_out(self.seed, nnodes, nodes,
                                                                          nmodules, module_mappings,
                                                                          conn)
            net_id = self.init_net_id(level, module=module)

            df_alpha = self.task_workflow(conn, w_in, output_nodes,
                                          readout_modules, net_id=net_id,
                                          compute_LE=self.compute_LE,
                                          sample_weight=sample_weight,
                                          multioutput=self.multioutput)
            scores[level] = df_alpha
        self.results = scores

        self.save_results()

    def hyperparameter_opt(self, sbms, nnodes, nodes,
                           nmodules, module_mappings, nseeds,
                           input_gain=None,
                           l2_alpha=None,
                           pruning_ratio=None):

        if input_gain is not None:
            self.input_gain = input_gain
        if l2_alpha is not None:
            self.l2_alpha = l2_alpha
        if pruning_ratio is not None:
            self.pruning_ratio = pruning_ratio

        self.readout = Readout(estimator=RidgeClassifier(alpha=self.l2_alpha,
                                                         fit_intercept=False))

        scores_seed = []
        for seed in range(nseeds):
            
            self.seed = seed + 2*len(sbms['1'])
            self.init_data(nseeds)
            sample_weight = self.sample_weight()

            scores_net_type = []
            for net_type in sbms.keys():

                conn = self.init_conn(seed, sbms, net_type)
                if conn is None:
                    return None
                
                module, output_nodes, readout_modules, w_in = self.set_in_out(seed, nnodes, nodes,
                                                                              nmodules, module_mappings,
                                                                              conn)
                df_alpha = self.task_workflow(conn, w_in, output_nodes, 
                                              readout_modules, 
                                              sample_weight=sample_weight)
                scores_net_type.append(df_alpha)
            scores_seed.append(scores_net_type)

        return scores_seed
    
class MemoryCapacityMultitasking(ABC):

    """
    Intermediate abstract class for Memory Capacity and Multitasking tasks

    Methods
    -------
    get_MC_data(seed)
        Generate Memory Capacity task data
    """

    def get_MC_data(self, seed):

        x, y = self.task.fetch_data(n_trials=self.duration,
                                    horizon_max=self.horizon_max,
                                    win=self.warmup, seed=seed)

        return x, y

class NonlinearTransformationMultitasking(ABC):

    """
    Intermediate abstract class for Nonlinear Transformation and 
    Multitasking tasks

    Methods
    -------
    get_NLT_data(ncycles, lag=False, rand=True, seed=0)
        Generate Nonlinear Transformation task data
    """

    def get_NLT_data(self, ncycles, lag=False, rand=True, seed=0):

        total_duration = self.total_duration + 1
        t = np.arange(total_duration)

        phase = 0
        #phase randomization
        if rand == True:
            np.random.seed(seed)
            phase = np.random.uniform(0, 2*np.pi)

        #angular conversion
        rad = 2*np.pi*ncycles*t/total_duration + phase

        x = np.sin(rad)[:, np.newaxis]
        x = x[1:]
        y = signal.square(rad)
        y = y[1:]

        if lag == True:
            y = x[self.warmup - 1: -1]
        else:
            y = x[self.warmup:]

        return x, y
    
class ChaoticPrediction(Task, RegressionUtils):

    """
    Class for running the Chaotic Prediction task

    Attributes
    ----------
    horizon : int
        Horizon for the task
    task_name : str
        Name of the task
    task : ReservoirPyTask
        Task object for the task
    input_amp : bool
        Flag for amplifying the input signal
    distributed_input : bool
        Flag for distributing the input signal
    min : float
        Minimum value for the initial conditions
    max : float
        Maximum value for the initial conditions
    data_kwargs : dict
        Additional keyword arguments for the task
    training_noise : bool
        Flag for adding noise to the training data
    noise_factor : float
        Noise factor
    test_split : int
        Test split duration
    x_train : np.ndarray
        Training input data
    y_train : np.ndarray
        Training output data
    x_test : np.ndarray
        Testing input data
    y_test : np.ndarray
        Testing output data

    Methods
    -------
    init_conds(seed)
        Initialize initial conditions for the task
    init_data(nseeds)
        Initialize Chaotic Prediction task data
    set_in_out(seed, nnodes, nodes, nmodules, module_mappings, conn)
        Set input and output nodes
    """

    def __init__(self, config, **kwargs):

        super().__init__(config)

        self.horizon = config.horizon
        self.task_name = config.task_name
        self.task = ReservoirPyTask(name=self.task_name)
        self.input_amp = config.input_amp
        self.distributed_input = config.distributed_input
        self.min = config.init_min
        self.max = config.init_max
        self.data_kwargs = kwargs
        self.training_noise = config.training_noise
        self.noise_factor = config.noise_factor
        self.test_split = config.test_split

    def init_conds(self, seed):

        np.random.seed(seed)
        if self.task_name == 'henon_map':
            x0 = np.random.uniform(self.min, self.max, 2)
        elif self.task_name == 'logistic_map':
            x0 = np.random.uniform(self.min, self.max, 1)
        elif self.task_name == 'lorenz':
            x0 = np.random.uniform(self.min, self.max, 3)
        elif self.task_name == 'mackey_glass':
            x0 = np.random.uniform(self.min, self.max, 1)
        elif self.task_name == 'multiscroll':
            x0 = np.random.uniform(self.min, self.max, 3)
        elif self.task_name == 'doublescroll':
            x0 = np.random.uniform(self.min, self.max, 3)
        elif self.task_name == 'rabinovich_fabrikant':
            x0 = np.random.uniform(self.min, self.max, 3)
        elif self.task_name == 'narma':
            x0 = np.random.uniform(self.min, self.max, 1)
        elif self.task_name == 'lorenz96':
            size = self.data_kwargs.get('N', 36)
            x0 = np.random.uniform(self.min, self.max, size)
        elif self.task_name == 'rossler':
            x0 = np.random.uniform(self.min, self.max, 3)

        return x0

    def init_data(self, nseeds):
        
        if 'x0' not in self.data_kwargs:
            x0 = self.init_conds(self.seed)
            self.data_kwargs['x0'] = x0
        if self.task_name == 'mackey_glass' or self.task_name == 'narma':
            if 'seed' not in self.data_kwargs:
                self.data_kwargs['seed'] = self.seed
        custom_u = False
        if self.task_name == 'narma':
            if 'u_min' in self.data_kwargs or 'u_max' in self.data_kwargs:
                custom_u = True
                u_min = 0 if 'u_min' not in self.data_kwargs else self.data_kwargs['u_min']
                u_max = 0.45 if 'u_max' not in self.data_kwargs else self.data_kwargs['u_max']
                order = 30 if 'order' not in self.data_kwargs else self.data_kwargs['order']
                #delete u_min and u_max from data_kwargs
                if 'u_min' in self.data_kwargs:
                    del self.data_kwargs['u_min']
                if 'u_max' in self.data_kwargs:
                    del self.data_kwargs['u_max']
                np.random.seed(self.seed)
                duration = self.duration + self.warmup + np.abs(self.horizon) + 1 + order
                u = np.random.uniform(u_min, u_max, duration)
                u = u[:, np.newaxis]
                self.data_kwargs['u'] = u
                
        self.x_train, self.y_train = self.task.fetch_data(n_trials=self.duration,
                                                          horizon=self.horizon,
                                                          win=self.warmup,
                                                          **self.data_kwargs)
        if self.training_noise:
            np.random.seed(self.seed)
            x_train_mean = np.mean(self.x_train)
            self.x_train += np.random.uniform(x_train_mean - self.noise_factor*x_train_mean, 
                                              x_train_mean + self.noise_factor*x_train_mean, 
                                              self.x_train.shape)
        if 'x0' not in self.data_kwargs:
            x0 = self.init_conds(self.seed + nseeds)
            self.data_kwargs['x0'] = x0
        if self.task_name == 'mackey_glass' or self.task_name == 'narma':
            if 'seed' not in self.data_kwargs:
                self.data_kwargs['seed'] = self.seed + nseeds
        if custom_u:
            np.random.seed(self.seed + nseeds)
            u = np.random.uniform(u_min, u_max, duration)
            u = u[:, np.newaxis]
            self.data_kwargs['u'] = u

        self.x_test, self.y_test = self.task.fetch_data(n_trials=self.duration,
                                                        horizon=self.horizon,
                                                        win=self.warmup,
                                                        **self.data_kwargs)

    def set_in_out(self, seed, nnodes, nodes,
                   nmodules, module_mappings, conn):

        np.random.seed(seed)
        w_in = np.zeros((self.task.n_features, conn.n_nodes))

        #make sure there are not more task features than modules
        if (self.task.n_features == 1 or 
            (self.distributed_input and self.task.n_features < nmodules)):

            #select the input modules
            input_modules = np.random.choice(range(nmodules), self.task.n_features, replace=False)
            #all other modules are output modules
            output_modules = np.array([i for i in range(nmodules) if i not in input_modules])

            input_nodes = []
            for input_module in input_modules:
                potential_input_nodes = np.array(range(input_module*nnodes, 
                                                       input_module*nnodes + 
                                                       nnodes))
                if self.input_amp:
                    input_nodes.append(potential_input_nodes)
                #select a random node in each input module if not amplifying
                else:
                    input_node = conn.get_nodes(
                        'random', nodes_from=potential_input_nodes, seed=seed
                    )
                    input_nodes.append(input_node)

            output_nodes = []
            readout_modules = []
            for output_module in output_modules:
                curr_output_nodes = np.array(range(output_module*nnodes, 
                                                   output_module*nnodes + 
                                                   nnodes))
                output_nodes.append(curr_output_nodes)
                readout_modules.append(module_mappings[curr_output_nodes])
            output_nodes = np.concatenate(output_nodes)

            #map the input signals to the input nodes
            for i in range(self.task.n_features):
                w_in[i, input_nodes[i]] = 1
            
            if self.task.n_features == 1:
                module = input_modules[0]
            else:
                module = None

        else:
            #warn that distributed input is not possible
            if self.distributed_input:
                print("Distributed input is not possible for this task.")

            #select a random module
            module = np.random.randint(nmodules)

            #select the input nodes from the module
            potential_input_nodes = np.array(range(module*nnodes, 
                                                   module*nnodes + 
                                                   nnodes))
            input_nodes = conn.get_nodes(
                    'random', nodes_from=potential_input_nodes,
                    n_nodes=self.task.n_features, seed=seed
                )
            
            #all other modules are output modules
            output_nodes = (nodes < module*nnodes)|(nodes >= module*nnodes+nnodes)
            readout_modules = module_mappings[output_nodes]

            #map the input signals to the input nodes
            w_in[:, input_nodes] = np.eye(self.task.n_features)

        return module, output_nodes, readout_modules, w_in
        
class NeurogymTask(Task, ClassifierUtils):

    """
    Class for running Neurogym tasks

    Attributes
    ----------
    task_name : str
        Name of the task
    task : NeurogymTask
        Task object for the task
    input_amp : bool
        Flag for amplifying the input signal
    distributed_input : bool
        Flag for distributing the input signal
    sample_weight_strat : str
        Sample weight strategy
    grace_period : int
        Grace period before evaluating
    training_noise : bool
        Flag for adding noise to the training data
    testing_noise : bool
        Flag for adding noise to the testing data
    max_noise : float
        Maximum noise amplitude
    data_kwargs : dict
        Additional keyword arguments for the task
    save_io_data_path : str
        Path to save input and output data
    load_io_data_path : str
        Path to load input and output data
    x_train : np.ndarray
        Training input data
    y_train : np.ndarray
        Training output data
    x_test : np.ndarray
        Testing input data
    y_test : np.ndarray
        Testing output data

    Methods
    -------
    init_data(nseeds)
        Initialize Neurogym task data
    save_io_data(nseeds)
        Save input and output data
    set_in_out(seed, nnodes, nodes, nmodules, module_mappings, conn)
        Set input and output nodes
    """

    def __init__(self, config, **kwargs):

        super().__init__(config)

        self.task_name = config.task_name
        self.task = NeuroGymTask(name=self.task_name)
        self.input_amp = config.input_amp
        self.distributed_input = config.distributed_input
        self.sample_weight_strat = config.sample_weight_strat
        self.grace_period = config.grace_period
        self.training_noise = config.training_noise
        self.testing_noise = config.testing_noise
        self.max_noise = config.max_noise
        self.data_kwargs = kwargs
        self.save_io_data_path = config.save_io_data_path
        self.load_io_data_path = config.load_io_data_path

    def init_data(self, nseeds):

        if self.load_io_data_path is not None:
            with open(os.path.join(self.load_io_data_path,
                                   'x_train_seed{}.pickle'.format(self.seed)), 'rb') as f:
                self.x_train = pickle.load(f)
            with open(os.path.join(self.load_io_data_path,
                                   'y_train_seed{}.pickle'.format(self.seed)), 'rb') as f:
                self.y_train = pickle.load(f)
            with open(os.path.join(self.load_io_data_path,
                                   'x_test_seed{}.pickle'.format(self.seed + nseeds)), 'rb') as f:
                self.x_test = pickle.load(f)
            with open(os.path.join(self.load_io_data_path,
                                   'y_test_seed{}.pickle'.format(self.seed + nseeds)), 'rb') as f:
                self.y_test = pickle.load(f)
        #was already saved for this experiment
        elif os.path.exists(os.path.join(self.save_io_data_path, 'x_train_seed{}.pickle'.format(self.seed))):
            with open(os.path.join(self.save_io_data_path,
                                   'x_train_seed{}.pickle'.format(self.seed)), 'rb') as f:
                 self.x_train = pickle.load(f)
            with open(os.path.join(self.save_io_data_path,
                                   'y_train_seed{}.pickle'.format(self.seed)), 'rb') as f:
                  self.y_train = pickle.load(f)
            with open(os.path.join(self.save_io_data_path,
                                   'x_test_seed{}.pickle'.format(self.seed + nseeds)), 'rb') as f:
                 self.x_test = pickle.load(f)  
            with open(os.path.join(self.save_io_data_path,
                                   'y_test_seed{}.pickle'.format(self.seed + nseeds)), 'rb') as f:
                  self.y_test = pickle.load(f)
        else:
            self.x_train, self.y_train = self.task.fetch_data(n_trials=self.duration, **self.data_kwargs)
            if self.training_noise:
                np.random.seed(self.seed)
                for trial in range(len(self.x_train)):
                    self.x_train[trial] += np.random.uniform(-self.max_noise, self.max_noise, self.x_train[trial].shape)
            self.x_test, self.y_test = self.task.fetch_data(n_trials=self.duration, **self.data_kwargs)
            if self.testing_noise:
                np.random.seed(self.seed + nseeds)
                for trial in range(len(self.x_test)):
                    self.x_test[trial] += np.random.uniform(-self.max_noise, self.max_noise, self.x_test[trial].shape)
            self.save_io_data(nseeds)
    
    def save_io_data(self, nseeds):

        with open(os.path.join(self.save_io_data_path,
                               'x_train_seed{}.pickle'.format(self.seed)), 'wb') as f:
            pickle.dump(self.x_train, f)
        with open(os.path.join(self.save_io_data_path,
                               'y_train_seed{}.pickle'.format(self.seed)), 'wb') as f:
            pickle.dump(self.y_train, f)
        with open(os.path.join(self.save_io_data_path,
                               'x_test_seed{}.pickle'.format(self.seed + nseeds)), 'wb') as f:
            pickle.dump(self.x_test, f)
        with open(os.path.join(self.save_io_data_path,
                               'y_test_seed{}.pickle'.format(self.seed + nseeds)), 'wb') as f:
            pickle.dump(self.y_test, f)

    def set_in_out(self, seed, nnodes, nodes,
                   nmodules, module_mappings, conn):

        np.random.seed(seed)
        w_in = np.zeros((self.task.n_features, conn.n_nodes))

        #make sure there are not more task features than modules
        if (self.task.n_features == 1 or 
            (self.distributed_input and self.task.n_features < nmodules)):

            #select the input modules
            input_modules = np.random.choice(range(nmodules), self.task.n_features, replace=False)
            #all other modules are output modules
            output_modules = np.array([i for i in range(nmodules) if i not in input_modules])

            input_nodes = []
            for input_module in input_modules:
                potential_input_nodes = np.array(range(input_module*nnodes, 
                                                       input_module*nnodes + 
                                                       nnodes))
                if self.input_amp:
                    input_nodes.append(potential_input_nodes)
                #select a random node in each input module if not amplifying
                else:
                    input_node = conn.get_nodes(
                        'random', nodes_from=potential_input_nodes, seed=seed
                    )
                    input_nodes.append(input_node)

            output_nodes = []
            readout_modules = []
            for output_module in output_modules:
                curr_output_nodes = np.array(range(output_module*nnodes, 
                                                   output_module*nnodes + 
                                                   nnodes))
                output_nodes.append(curr_output_nodes)
                readout_modules.append(module_mappings[curr_output_nodes])
            output_nodes = np.concatenate(output_nodes)

            #map the input signals to the input nodes
            for i in range(self.task.n_features):
                w_in[i, input_nodes[i]] = 1
            module = None

        else:
            #warn that distributed input is not possible
            if self.distributed_input:
                print("Distributed input is not possible for this task.")

            #select a random module
            module = np.random.randint(nmodules)

            #select the input nodes from the module
            potential_input_nodes = np.array(range(module*nnodes, 
                                                   module*nnodes + 
                                                   nnodes))
            input_nodes = conn.get_nodes(
                    'random', nodes_from=potential_input_nodes,
                    n_nodes=self.task.n_features, seed=seed
                )
            
            #all other modules are output modules
            output_nodes = (nodes < module*nnodes)|(nodes >= module*nnodes+nnodes)
            readout_modules = module_mappings[output_nodes]

            #map the input signals to the input nodes
            w_in[:, input_nodes] = np.eye(self.task.n_features)

        return module, output_nodes, readout_modules, w_in

class MemoryCapacity(Task, RegressionUtils, MemoryCapacityMultitasking):

    """
    Class for running the Memory Capacity task

    Attributes
    ----------
    horizon_max : int
        Maximum time-lag for the task
    task : Conn2ResTask
        Task object for the task
    input_amp : bool
        Flag for amplifying the input signal
    training_noise : bool
        Flag for adding noise to the training data
    testing_noise : bool
        Flag for adding noise to the testing data
    max_noise : float
        Maximum noise amplitude
    x_train : np.ndarray
        Training input data
    y_train : np.ndarray
        Training output data
    x_test : np.ndarray
        Testing input data
    y_test : np.ndarray
        Testing output data

    Methods
    -------
    init_data(nseeds)
        Initialize Memory Capacity task data
    """

    def __init__(self, config):

        super().__init__(config)

        self.horizon_max = config.horizon_max
        self.task = Conn2ResTask(name='MemoryCapacity')
        self.input_amp = config.input_amp
        self.training_noise = config.training_noise
        self.testing_noise = config.testing_noise
        self.max_noise = config.max_noise
        
    def init_data(self, nseeds):

        self.x_train, self.y_train = self.get_MC_data(self.seed)
        if self.training_noise:
            np.random.seed(self.seed)
            self.x_train += np.random.uniform(-self.max_noise, self.max_noise, 
                                              self.x_train.shape)
        self.x_test, self.y_test = self.get_MC_data(self.seed + nseeds)
        if self.testing_noise:
            np.random.seed(self.seed + nseeds)
            self.x_test += np.random.uniform(-self.max_noise, self.max_noise, 
                                             self.x_test.shape)

class EmpiricalMC(MemoryCapacity):

    """
    Class for running the Memory Capacity task
    on empirical connectivity matrices

    Attributes
    ----------
    seed : int
        Random seed for the task
    readout : Readout
        Readout object for the task
    results : dict
        Performance results for the task
    input_gain : float
        Input gain for the reservoir
    l2_alpha : float
        Ridge regularization parameter

    Methods
    -------
    set_in_out(seed, conn, module, module_mappings, noutputs)
        Set input and output nodes for empirical connectivity matrices
    analysis(nets, module_mappings, noutputs, seed)
        Run the Memory Capacity task on empirical connectivity matrices
    hyperparameter_opt(nets, module_mappings, noutputs,
                       niter, nseeds, sampler_seed,
                       gain_extrema=None, ridge_extrema=None,
                       pruning_extrema=None)
        Run hyperparameter optimization for the Memory Capacity task
        on empirical connectivity matrices
    best_hyperparameter(niter, nseeds, aggregate='average',
                        gain_opt=False, ridge_opt=False,
                        density_opt=False)
        Return the best empirical hyperparameters
    """

    def __init__(self, config):

        super().__init__(config)

    def set_in_out(self, seed, conn,
                   module, module_mappings, noutputs):

        input_nodes = np.where(module_mappings == module)[0]
        potential_output_nodes = np.where(module_mappings != module)[0]
        output_modules = np.unique(module_mappings[potential_output_nodes])
        output_nodes = []
        #picking the same number of output nodes for each output module
        #to ensure a similar dimensionality expansion
        for output_module in output_modules:
            curr_output_nodes = conn.get_nodes('random', nodes_from=np.where(module_mappings == output_module)[0],
                                                n_nodes=noutputs, seed=seed)
            output_nodes.append(curr_output_nodes)
        output_nodes = np.concatenate(output_nodes)
        readout_modules = module_mappings[output_nodes]

        w_in = np.zeros((1, conn.n_nodes))
        w_in[:, input_nodes] = 1

        return output_nodes, readout_modules, w_in

    def analysis(self, nets, module_mappings, noutputs, seed):

        print('Running seed {}'.format(seed))
        self.seed = seed
        self.init_data(len(list(nets.values())[-1]))
        self.readout = Readout(estimator=Ridge(alpha=self.l2_alpha,
                                               fit_intercept=False))

        MCs = {}
        for level in nets.keys():

            if level == 'empirical' and self.seed > 0:
                conn = self.init_conn(0, nets, level)
            else:
                conn = self.init_conn(self.seed, nets, level)

            MC_modules = []
            #have to all be looped because of size and connectivity variability
            for module in np.unique(module_mappings):

                output_nodes, readout_modules, w_in = self.set_in_out(self.seed, conn,
                                                                      module, module_mappings,
                                                                      noutputs)
                net_id = self.init_net_id(level, module=module)
                df_alpha = self.task_workflow(conn, w_in, output_nodes,
                                              readout_modules, net_id=net_id,
                                              compute_LE=self.compute_LE,
                                              multioutput=self.multioutput)
                MC_modules.append(df_alpha)
            MCs[level] = MC_modules

        self.results = MCs
        self.save_results()

    def hyperparameter_opt(self, nets, module_mappings, 
                           noutputs, nseeds,
                           input_gain=None,
                           l2_alpha=None,
                           pruning_ratio=None):
        
        if input_gain is not None:
            self.input_gain = input_gain
        if l2_alpha is not None:
            self.l2_alpha = l2_alpha

        self.readout = Readout(estimator=Ridge(alpha=self.l2_alpha,
                                               fit_intercept=False))
        conn = self.init_conn(0, nets, 'empirical')
        
        MC_seed = []
        for seed in range(nseeds):

            self.seed = seed + 2*len(list(nets.values())[-1])
            self.init_data(nseeds)

            MC_modules = []
            for module in np.unique(module_mappings):

                output_nodes, readout_modules, w_in = self.set_in_out(seed, conn,
                                                                      module, module_mappings,
                                                                      noutputs)
                df_alpha = self.task_workflow(conn, w_in, output_nodes, 
                                              readout_modules)
                MC_modules.append(df_alpha)
            MC_seed.append(MC_modules)

        return MC_seed

    #scores are averaged or maxed across output and input modules
    #maxed across alpha values
    #averaged across seeds
    #finally, the highest performing parameter is chosen
    def best_hyperparameter(self, niter, nseeds,
                            aggregate='average',
                            gain_opt=False, ridge_opt=False,
                            density_opt=False):

        input_gain = None
        l2_alpha = None
        pruning_ratio = None

        scores = []
        for MC_niter in self.hyperparameter_results:
            agg_scores = []
            for MC_seed in MC_niter:
                max_scores = []
                for MC in MC_seed:
                    #aggregate scores across output modules
                    if aggregate == 'average':
                        df_agg = MC.groupby('alpha').agg({self.score: 'mean'}).reset_index()
                    elif aggregate == 'max':
                        df_agg = MC.groupby('alpha').agg({self.score: 'max'}).reset_index()
                    else:
                        raise ValueError("Invalid aggregation method. "\
                                         "Choose from 'average' or 'max'.")
                    #max score across alpha values
                    max_scores.append(df_agg[self.score].max())
                #aggregate scores across input modules
                if aggregate == 'average':
                    agg_scores.append(np.mean(max_scores))
                elif aggregate == 'max':
                    agg_scores.append(np.max(max_scores))
                else:
                    raise ValueError("Invalid aggregation method. "\
                                     "Choose from 'average' or 'max'.")
            #mean score across seeds
            scores.append(np.mean(agg_scores))
        #best hyperparameter
        if gain_opt:
            self.input_gain = self.param_sampler[np.argmax(scores)]['input_gain']
            input_gain = self.input_gain
        if ridge_opt:
            self.l2_alpha = self.param_sampler[np.argmax(scores)]['l2_alpha']
            l2_alpha = self.l2_alpha

        return input_gain, l2_alpha, pruning_ratio

class NLT(Task, RegressionUtils, NonlinearTransformationMultitasking):

    """
    Class for running the Nonlinear Transformation task

    Attributes
    ----------
    ncycles : int
        Number of cycles for the task
    lag : bool
        Whether to use the lagged version of the task
    input_amp : bool
        Flag for amplifying the input signal
    x_train : np.ndarray
        Training input data
    y_train : np.ndarray
        Training output data
    x_test : np.ndarray
        Testing input data
    y_test : np.ndarray
        Testing output data

    Methods
    -------
    init_NLT_data(nseeds)
        Initialize Nonlinear Transformation task data
    """

    def __init__(self, config):

        super().__init__(config)

        self.ncycles = config.ncycles
        self.lag = config.lag
        self.input_amp = config.input_amp

    def init_data(self, nseeds):

        self.x_train, self.y_train = self.get_NLT_data(self.ncycles, lag=self.lag, seed=self.seed)
        self.x_test, self.y_test = self.get_NLT_data(self.ncycles, lag=self.lag, seed=self.seed + nseeds)

class Multitasking(Task, MemoryCapacityMultitasking, NonlinearTransformationMultitasking):

    """
    Class for running the Multitasking task

    Attributes
    ----------
    ninputs : int
        Number of input signals
    interleaved : bool
        Whether the tasks are interleaved
    ncycles1 : int
        Number of cycles for the first task
    ncycles2 : int
        Number of cycles for the second task
    ncycles3 : int
        Number of cycles for the third task
    ncycles4 : int
        Number of cycles for the fourth task
    lag : bool
        Whether to use the lagged version of the Nonlinear Transformation task
    horizon_max : int
        Maximum time-lag for the Memory Capacity task
    task : Conn2ResTask
        Task object for the Memory Capacity task
    training_noise : bool
        Flag for adding noise to the training data
    testing_noise : bool
        Flag for adding noise to the testing data
    max_noise : float
        Maximum noise amplitude
    x_train : np.ndarray
        Training input data
    y_train : np.ndarray
        Training output data
    x_test : np.ndarray
        Testing input data
    y_test : np.ndarray
        Testing output data
    seed : int
        Random seed for the task
    readout : Readout
        Readout object for the task
    results : dict
        Performance results for the task
    input_gain : float
        Input gain for the reservoir
    l2_alpha : float
        Ridge regularization parameter
    pruning_ratio : float
        Ratio of connections to prune

    Methods
    -------
    init_multitasking_data(nseeds)
        Initialize Multitasking task data
    set_in_out(seed, conn, nnodes, nmodules, module_mappings)
        Set input and output nodes for the Multitasking task
    task_workflow(conn, w_in, nnodes, output_nodes, nmodules, readout_modules,
             net_id=None, compute_LE=False, sample_weight=None, 
             multioutput='uniform_average')
        Run the Multitasking task
    analysis(sbms, nnodes, nodes, nmodules, module_mappings, seed)
        Run a Multitasking analysis
    hyperparameter_opt(sbms, nnodes, nodes, nmodules, module_mappings,
                       nseeds, input_gain=None, l2_alpha=None,
                       pruning_ratio=None)
        Run hyperparameter optimization for the Multitasking task
    best_hyperparameter(niter, nseeds,
                        gain_opt=False, ridge_opt=False,
                        density_opt=False)
        Return the best multitasking hyperparameters
    """

    def __init__(self, config):

        super().__init__(config)

        self.ninputs = config.ninputs
        self.interleaved = config.interleaved

        self.ncycles1 = config.ncycles[0]
        if self.ninputs == 4:
            self.ncycles2 = config.ncycles[1]
        elif self.ninputs == 8:
            self.ncycles2 = config.ncycles[1]
            self.ncycles3 = config.ncycles[2]
            self.ncycles4 = config.ncycles[3]
        
        self.lag = config.lag

        self.horizon_max = config.horizon_max
        self.task = Conn2ResTask(name='MemoryCapacity')

        self.training_noise = config.training_noise
        self.testing_noise = config.testing_noise
        self.max_noise = config.max_noise

    def init_multitasking_data(self, nseeds):

        x1_train, y1_train = self.get_MC_data(self.seed)
        x1_test, y1_test = self.get_MC_data(self.seed + nseeds)

        if self.ninputs == 4:
            x2_train, y2_train = self.get_MC_data(self.seed + 2*nseeds)
            x2_test, y2_test = self.get_MC_data(self.seed + 3*nseeds)
        elif self.ninputs == 8:
            x2_train, y2_train = self.get_MC_data(self.seed + 2*nseeds)
            x2_test, y2_test = self.get_MC_data(self.seed + 3*nseeds)
            x3_train, y3_train = self.get_MC_data(self.seed + 4*nseeds)
            x3_test, y3_test = self.get_MC_data(self.seed + 5*nseeds)
            x4_train, y4_train = self.get_MC_data(self.seed + 6*nseeds)
            x4_test, y4_test = self.get_MC_data(self.seed + 7*nseeds)

        x5_train, y5_train = self.get_NLT_data(self.ncycles1, lag=self.lag, seed=self.seed)
        x5_test, y5_test = self.get_NLT_data(self.ncycles1, lag=self.lag, seed=self.seed + nseeds)

        if self.ninputs == 4:
            x6_train, y6_train = self.get_NLT_data(self.ncycles2, lag=self.lag, seed=self.seed + 2*nseeds)
            x6_test, y6_test = self.get_NLT_data(self.ncycles2, lag=self.lag, seed=self.seed + 3*nseeds)
        elif self.ninputs == 8:
            x6_train, y6_train = self.get_NLT_data(self.ncycles2, lag=self.lag, seed=self.seed + 2*nseeds)
            x6_test, y6_test = self.get_NLT_data(self.ncycles2, lag=self.lag, seed=self.seed + 3*nseeds)
            x7_train, y7_train = self.get_NLT_data(self.ncycles3, lag=self.lag, seed=self.seed + 4*nseeds)
            x7_test, y7_test = self.get_NLT_data(self.ncycles3, lag=self.lag, seed=self.seed + 5*nseeds)
            x8_train, y8_train = self.get_NLT_data(self.ncycles4, lag=self.lag, seed=self.seed + 6*nseeds)
            x8_test, y8_test = self.get_NLT_data(self.ncycles4, lag=self.lag, seed=self.seed + 7*nseeds)

        if self.ninputs == 2:
            self.x_train = np.hstack((x1_train, x5_train))
            self.x_test = np.hstack((x1_test, x5_test))

            self.y_train = [y1_train, y5_train]
            self.y_test = [y1_test, y5_test]
        elif self.ninputs == 4:
            if self.interleaved:
                self.x_train = np.hstack((x1_train, x5_train, x2_train, x6_train))
                self.x_test = np.hstack((x1_test, x5_test, x2_test, x6_test))
                self.y_train = [y1_train, y5_train, y2_train, y6_train]
                self.y_test = [y1_test, y5_test, y2_test, y6_test]
            else:
                self.x_train = np.hstack((x1_train, x2_train, x5_train, x6_train))
                self.x_test = np.hstack((x1_test, x2_test, x5_test, x6_test))
                self.y_train = [y1_train, y2_train, y5_train, y6_train]
                self.y_test = [y1_test, y2_test, y5_test, y6_test]
        elif self.ninputs == 8:
            if self.interleaved:
                self.x_train = np.hstack((x1_train, x5_train, x2_train, x6_train,
                                          x3_train, x7_train, x4_train, x8_train))
                self.x_test = np.hstack((x1_test, x5_test, x2_test, x6_test,
                                         x3_test, x7_test, x4_test, x8_test))
                self.y_train = [y1_train, y5_train, y2_train, y6_train,
                                y3_train, y7_train, y4_train, y8_train]
                self.y_test = [y1_test, y5_test, y2_test, y6_test,
                               y3_test, y7_test, y4_test, y8_test]
            else:
                self.x_train = np.hstack((x1_train, x2_train, x3_train, x4_train,
                                        x5_train, x6_train, x7_train, x8_train))
                self.x_test = np.hstack((x1_test, x2_test, x3_test, x4_test,
                                        x5_test, x6_test, x7_test, x8_test))

                self.y_train = [y1_train, y2_train, y3_train, y4_train,
                                y5_train, y6_train, y7_train, y8_train]
                self.y_test = [y1_test, y2_test, y3_test, y4_test,
                            y5_test, y6_test, y7_test, y8_test]
            
        if self.training_noise:
            np.random.seed(self.seed)
            for trial in range(len(self.x_train)):
                self.x_train[trial] += np.random.uniform(-self.max_noise, self.max_noise, self.x_train[trial].shape)
        if self.testing_noise:
            np.random.seed(self.seed + nseeds)
            for trial in range(len(self.x_test)):
                self.x_test[trial] += np.random.uniform(-self.max_noise, self.max_noise, self.x_test[trial].shape)

    def set_in_out(self, seed, conn, nnodes, nmodules, module_mappings):

        #one single input node per module
        #all other nodes are output nodes
        if self.ninputs == 8:
            input_nodes = []
            output_nodes = []
            readout_modules = []
            for input in range(self.ninputs):
                module_nodes = np.array(range(input*nnodes,
                                              input*nnodes + nnodes))
                curr_input_node = conn.get_nodes('random',
                                                 nodes_from=module_nodes,
                                                 seed=seed)
                input_nodes.append(curr_input_node)
                curr_output_nodes = module_nodes[np.where(module_nodes !=
                                                          curr_input_node)]
                output_nodes.append(curr_output_nodes)
                readout_modules.append(module_mappings[curr_output_nodes])
            output_nodes = np.concatenate(output_nodes)

        #two modules in different higher-order modules as input
        #all other modules are output modules
        elif self.ninputs == 2:
            np.random.seed(seed)
            seed1 = np.random.randint(0, nmodules//2)
            seed2 = np.random.randint(nmodules//2, nmodules)
            input_nodes_1 = np.array(range(seed1*nnodes,
                                           seed1*nnodes + nnodes))
            input_nodes_2 = np.array(range(seed2*nnodes,
                                           seed2*nnodes + nnodes))
            input_nodes = [input_nodes_1, input_nodes_2]
            output_nodes = []
            readout_modules = []
            for module in range(nmodules):
                if module != seed1 and module != seed2:
                    curr_output_nodes = np.array(range(module*nnodes,
                                                       module*nnodes + nnodes))
                    output_nodes.append(curr_output_nodes)
                    readout_modules.append(module_mappings[curr_output_nodes])
            output_nodes = np.concatenate(output_nodes)
        
        #input modules are even-numbered
        #output modules are odd-numbered
        elif self.ninputs == 4:
            input_nodes = []
            output_nodes = []
            readout_modules = []
            for input in range(self.ninputs*2):
                module_nodes = np.array(range(input*nnodes,
                                              input*nnodes + nnodes))
                if input % 2 == 0:
                    input_nodes.append(module_nodes)
                else:
                    output_nodes.append(module_nodes)
                    readout_modules.append(module_mappings[module_nodes])

            output_nodes = np.concatenate(output_nodes)

        w_in = np.zeros((self.ninputs, conn.n_nodes))
        for i in range(self.ninputs):
            w_in[i, input_nodes[i]] = 1

        return output_nodes, readout_modules, w_in

    def task_workflow(self, conn, w_in, nnodes, output_nodes,
                      nmodules, readout_modules, net_id=None,
                      compute_LE=False, multioutput='uniform_average'):

        if self.ninputs == 8:
            nnodes -= 1
            noutput_modules = range(nmodules)
        elif self.ninputs == 2:
            noutput_modules = range(nmodules - self.ninputs)
        elif self.ninputs == 4:
            noutput_modules = range(nmodules - self.ninputs)


        df_alpha = []
        for alpha in self.alphas:

            esn, rs_train, rs_test = self.simulation(alpha, conn, w_in,
                                                     output_nodes, compute_LE, 
                                                     multioutput, 'MT')
            if net_id is not None:
                self.save_rs(esn, alpha, net_id)

            if compute_LE:
                self.save_LEs(esn, alpha, net_id)

            for module in noutput_modules:

                curr_rs_train = rs_train[:, module*nnodes:(module + 1)*nnodes]
                curr_rs_test = rs_test[:, module*nnodes:(module + 1)*nnodes]

                if self.ninputs == 8 or self.ninputs == 4:
                    curr_y_train, curr_y_test = self.y_train[module], self.y_test[module]
                elif self.ninputs == 2:
                    y_id = 0 if module < len(noutput_modules)//2 else 1
                    curr_y_train, curr_y_test = self.y_train[y_id], self.y_test[y_id]
                

                df_res = self.readout.run_task(
                    X=(curr_rs_train, curr_rs_test), 
                    y=(curr_y_train, curr_y_test),
                    metric=self.score, 
                    readout_modules=readout_modules[module],
                    multioutput=multioutput
                )

                df_res['alpha'] = alpha
                df_alpha.append(df_res)
        full_df_alpha = pd.concat(df_alpha, ignore_index=True)
        avg_df_alpha = full_df_alpha.groupby('alpha')[self.score].mean()

        return full_df_alpha, avg_df_alpha

    def analysis(self, sbms, nnodes, nodes,
                 nmodules, module_mappings,
                 seed):

        print('Running seed {}'.format(seed))
        self.seed = seed
        self.init_multitasking_data(len(sbms['1']))
        self.readout = Readout(estimator=Ridge(alpha=self.l2_alpha,
                                               fit_intercept=False))
        full_scores = {}
        avg_scores = {}
        for level in sbms.keys():

            conn = self.init_conn(self.seed, sbms, level)
            output_nodes, readout_modules, w_in = self.set_in_out(seed, conn, nnodes,
                                                                  nmodules, module_mappings)
            net_id = self.init_net_id(level)
            full_df_alpha, avg_df_alpha = self.task_workflow(conn, w_in, nnodes, output_nodes,
                                                             nmodules, readout_modules,
                                                             net_id=net_id, compute_LE=self.compute_LE,
                                                             multioutput=self.multioutput)
            full_scores[level] = full_df_alpha
            avg_scores[level] = avg_df_alpha

        self.results = (full_scores, avg_scores)
        self.save_results()

    def hyperparameter_opt(self, sbms, nnodes, nodes,
                           nmodules, module_mappings,
                           nseeds, input_gain=None,
                           l2_alpha=None,
                           pruning_ratio=None):

        if input_gain is not None:
            self.input_gain = input_gain
        if l2_alpha is not None:
            self.l2_alpha = l2_alpha
        if pruning_ratio is not None:
            self.pruning_ratio = pruning_ratio

        self.readout = Readout(estimator=Ridge(alpha=self.l2_alpha,
                                               fit_intercept=False))

        scores_seed = []
        for seed in range(nseeds):

            self.seed = seed + self.ninputs*len(sbms['1'])
            self.init_multitasking_data(nseeds)

            scores_net_type = []
            for net_type in sbms.keys():

                conn = self.init_conn(seed, sbms, net_type)
                if conn is None:
                    return None

                output_nodes, readout_modules, w_in = self.set_in_out(seed, conn, nnodes,
                                                                      nmodules, module_mappings)
                full_df_alpha, avg_df_alpha = self.task_workflow(conn, w_in, nnodes, output_nodes,
                                                                 nmodules, readout_modules)
                scores_net_type.append(avg_df_alpha)
            scores_seed.append(scores_net_type)
    
        return scores_seed

    #scores are maxed across alpha values
    #averaged across seeds
    #maxed across network types
    #finally, the highest performing parameter is chosen
    def best_hyperparameter(self, niter, nseeds,
                            aggregate='average',
                            gain_opt=False, ridge_opt=False,
                            density_opt=False):

        if aggregate != 'average':
            raise ValueError("Invalid aggregation method. "\
                             "Choose 'average' for multitasking.")

        input_gain = None
        l2_alpha = None
        pruning_ratio = None

        scores = []
        for scores_niter in self.hyperparameter_results:
            if scores_niter is None:
                scores.append(np.nan)
                continue
            scores_type = []
            for net_type in range(4):
                max_scores = []
                for seed in range(nseeds):
                    #max score across alpha values
                    max_scores.append(scores_niter[seed][net_type].values.max())
                #mean score across seeds
                scores_type.append(np.mean(max_scores))
            #max score across network types
            scores.append(np.max(scores_type))
        #best hyperparameter
        if gain_opt:
            self.input_gain = self.param_sampler[np.nanargmax(scores)]['input_gain']
            input_gain = self.input_gain
        if ridge_opt:
            self.l2_alpha = self.param_sampler[np.nanargmax(scores)]['l2_alpha']
            l2_alpha = self.l2_alpha
        if density_opt:
            self.pruning_ratio = self.param_sampler[np.nanargmax(scores)]['pruning_ratio']
            pruning_ratio = self.pruning_ratio

        return input_gain, l2_alpha, pruning_ratio