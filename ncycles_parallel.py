import pickle
import networkx as nx

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

from ncycles import compute_ncycles

experiment_path = 'results/SBM/SBM_MC_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse/'
network_data_path = experiment_path + 'network_data/'
nets = pickle.load(open(network_data_path + 'pruned_nets.pickle', 'rb'))

for net_type, net_arr in nets.items():
    n_cycles_list = Parallel(n_jobs=50, verbose=1)(delayed(compute_ncycles)(nx.from_numpy_array(net)) for net in net_arr)
    with open(network_data_path + f'ncycles_{net_type}.pickle', 'wb') as f:
        pickle.dump(n_cycles_list, f)
    print(f"Finished computing cycles for {net_type} networks.")