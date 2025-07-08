import bct
import numpy as np
import networkx as nx
from netneurotools.modularity import zrand
from netneurotools.metrics import communicability_wei
from randmio_und_hmod import randmio_und_hmod

import os
import pickle

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

class Network:

    """
    Base Network class

    Attributes
    ----------
    nnetworks : int
        Number of networks to generate
    net_path : str
        Path to save networks
    njobs : int
        Number of jobs to run in parallel

    Methods
    --------
    save_data(data, filename)
        Save data to a pickle file
    """

    def __init__(self, config):

        self.nnetworks = config.nnetworks
        self.net_path = config.net_path
        self.njobs = config.njobs

    def save_data(self, data, filename):

        with open(os.path.join(self.net_path, filename), 'wb') as f:
            pickle.dump(data, f)

class SBM(Network):

    """
    Stochastic Block Model network class

    Attributes
    ----------
    nnodes : int
        Number of nodes in each block
    delta_p : float
        Factor of decrease for connection probabilities
    nedges : int
        Number of edges in the network
    p1 : float
        Probability of connection within diagonal blocks
    min_weight : float
        Minimum weight for connections
    bin_directed : bool
        Whether to generate directed networks or not
    wei_directed : bool
        Whether the weights are directed/asymetric or not
    selfloops : bool
        Whether to include self-loops or not
    i_ratio : float
        Proportion of inhibitory connections
    p1_mat : list of lists
        Probability matrix for level 1
    p2_mat : list of lists
        Probability matrix for level 2
    p3_mat : list of lists
        Probability matrix for level 3
    nmodules : int
        Number of modules
    sizes : list
        Number of nodes in each module
    nodes : np.array
        Array of nodes
    module_mappings : np.array
        Array of module mappings for each node
    p1_nets : list of np.arrays
        Networks for level 1
    p2_nets : list of np.arrays
        Networks for level 2
    p3_nets : list of np.arrays
        Networks for level 3
    ms_nulls : list of np.arrays
        Networks for MS nulls
    ms_eff : list of floats
        Number of rewirings for MS nulls
    nets : dict
        Dictionary of all generated networks

    Methods
    -------
    generate_block_probability_matrices()
        Generate block probability matrices
    generate_networks(p_mat)
        Generate networks from a probability matrix
    get_module_mappings()
        Generate module mappings for each node
    generate_MS_nulls(networks, itr=10, seed=0)
        Generate MS null networks
    """

    def __init__(self, config):

        super().__init__(config)

        self.nnodes = config.nnodes
        self.delta_p = config.delta_p
        self.nedges = config.nedges
        if self.nedges is not None:
            self.p1 = self.nedges/(8*self.nnodes**2*(1 + self.delta_p + 
                                   2*self.delta_p**2 + 4*self.delta_p**3))
        else:
            self.p1 = config.p1
        self.min_weight = config.min_weight
        self.bin_directed = config.bin_directed
        self.wei_directed = config.wei_directed
        self.selfloops = config.selfloops
        self.i_ratio = config.i_ratio

        self.p1_mat, self.p2_mat, self.p3_mat = self.generate_block_probability_matrices()
        self.nmodules = len(self.p1_mat)
        self.sizes = [self.nnodes]*self.nmodules

        self.nodes = np.array(range(self.nnodes*self.nmodules))
        self.module_mappings = self.get_module_mappings()

        self.p1_nets = self.generate_networks(self.p1_mat)
        self.p2_nets = self.generate_networks(self.p2_mat)
        self.p3_nets = self.generate_networks(self.p3_mat)
        self.ms_nulls, self.ms_eff = self.generate_MS_nulls(self.p3_nets)

        self.nets = {'1': self.p1_nets, '2': self.p2_nets, '3': self.p3_nets, 
                     'MS': self.ms_nulls}

    def generate_block_probability_matrices(self):

        p1 = self.p1
        delta_p = self.delta_p
        p2 = p1*delta_p
        p3 = p2*delta_p
        p4 = p3*delta_p

        p3_mat = [[p1, p2, p3, p3, p4, p4, p4, p4],
                  [p2, p1, p3, p3, p4, p4, p4, p4],
                  [p3, p3, p1, p2, p4, p4, p4, p4],
                  [p3, p3, p2, p1, p4, p4, p4, p4],
                  [p4, p4, p4, p4, p1, p2, p3, p3],
                  [p4, p4, p4, p4, p2, p1, p3, p3],
                  [p4, p4, p4, p4, p3, p3, p1, p2],
                  [p4, p4, p4, p4, p3, p3, p2, p1]]

        #Average of merged p3 and p4 densities
        new_p3 = (2*p3 + 4*p4)/6
        p2_mat = [[p1, p2, new_p3, new_p3, new_p3, new_p3, new_p3, new_p3],
                  [p2, p1, new_p3, new_p3, new_p3, new_p3, new_p3, new_p3],
                  [new_p3, new_p3, p1, p2, new_p3, new_p3, new_p3, new_p3],
                  [new_p3, new_p3, p2, p1, new_p3, new_p3, new_p3, new_p3],
                  [new_p3, new_p3, new_p3, new_p3, p1, p2, new_p3, new_p3],
                  [new_p3, new_p3, new_p3, new_p3, p2, p1, new_p3, new_p3],
                  [new_p3, new_p3, new_p3, new_p3, new_p3, new_p3, p1, p2],
                  [new_p3, new_p3, new_p3, new_p3, new_p3, new_p3, p2, p1]]

        #Average of merged p2 and p3 densities
        new_p2 = (p2 + 6*new_p3)/7
        p1_mat = [[p1, new_p2, new_p2, new_p2, new_p2, new_p2, new_p2, new_p2],
                  [new_p2, p1, new_p2, new_p2, new_p2, new_p2, new_p2, new_p2],
                  [new_p2, new_p2, p1, new_p2, new_p2, new_p2, new_p2, new_p2],
                  [new_p2, new_p2, new_p2, p1, new_p2, new_p2, new_p2, new_p2],
                  [new_p2, new_p2, new_p2, new_p2, p1, new_p2, new_p2, new_p2],
                  [new_p2, new_p2, new_p2, new_p2, new_p2, p1, new_p2, new_p2],
                  [new_p2, new_p2, new_p2, new_p2, new_p2, new_p2, p1, new_p2],
                  [new_p2, new_p2, new_p2, new_p2, new_p2, new_p2, new_p2, p1]]

        return p1_mat, p2_mat, p3_mat

    def generate_networks(self, p_mat):

        networks = []
        for seed in range(self.nnetworks):
            #Generate binary SBM networks
            sbm = nx.stochastic_block_model(self.sizes, p_mat, seed=seed,
                                            directed=self.bin_directed,
                                            selfloops=self.selfloops)
            sbm = nx.to_numpy_array(sbm)
            #Assign weights
            np.random.seed(seed)
            if self.bin_directed == False and self.wei_directed == False:
                for i, j in list(zip(*np.nonzero(np.triu(sbm, k=1)))):
                    sbm[i, j] = np.random.uniform(low=self.min_weight)
                    sbm[j, i] = sbm[i, j]
                    if self.i_ratio > 0:
                        if self.module_mappings[i] == self.module_mappings[j]:
                            if np.random.uniform() < self.i_ratio:
                                sbm[i, j] *= -1
                                sbm[j, i] *= -1
                if self.selfloops:
                    for i in range(self.nnodes*self.nmodules):
                        sbm[i, i] = np.random.uniform(low=self.min_weight)
                    if self.i_ratio > 0:
                        if np.random.uniform() < self.i_ratio:
                            sbm[i, i] *= -1
            else:
                for i, j in list(zip(*np.nonzero(sbm))):
                    sbm[i, j] = np.random.uniform(low=self.min_weight)
                    if self.i_ratio > 0:
                        if self.module_mappings[i] == self.module_mappings[j]:
                            if np.random.uniform() < self.i_ratio:
                                sbm[i, j] *= -1

            networks.append(sbm)

        return networks

    def get_module_mappings(self):
        return np.repeat(np.arange(self.nmodules), self.nnodes)

    def generate_MS_nulls(self, networks, itr=10, seed=0):

        func = bct.randmio_und_connected if not self.bin_directed and not self.wei_directed else bct.randmio_dir_connected
        nulls = Parallel(n_jobs=self.njobs)(delayed(func)(network, itr=itr, seed=seed) for network in networks)
        nulls_arr = list(zip(*nulls))
        nulls = nulls_arr[0]
        nulls_eff = nulls_arr[1]

        return nulls, nulls_eff

class Empirical(Network):

    """
    Empirical network class

    Attributes
    ----------
    w : np.array
        Empirical network adjacency matrix
    gamma : float
        Resolution parameter for modularity maximization
    nseeds : int
        Number of seeds for modularity maximization
    zrands : np.array
        Sum of zrand scores for each community assignment
    max_zrand_idx : int
        Community assignment with the highest zrand score
    module_mappings : np.array
        Community assignments for each node at level 1
    module_mappings_level2 : np.array
        Community assignments for each node at level 2
    strength : np.array
        Node strengths
    mod_strength : dict
        Average strength for each module
    clustering : np.array
        Node clustering coefficients
    mod_clustering : dict
        Average clustering coefficient for each module
    participation : np.array
        Node participation coefficients
    mod_participation : dict
        Average participation coefficient for each module
    mod_nulls : list of np.arrays
        Modular null networks
    mod_nulls_eff : list of floats
        Number of rewirings for modular null networks
    hierarchical_mod_nulls : list of np.arrays
        Hierarchical modular null networks
    hierarchical_mod_nulls_eff : list of floats
        Number of rewirings for hierarchical modular null networks
    ms_nulls : list of np.arrays
        Degree-preserving null networks
    ms_nulls_eff : list of floats
        Number of rewirings for MS nulls
    nets : dict
        Dictionary of all generated networks

    Methods
    -------
    find_modules()
        Find community assignments using modularity maximization
    run_modularity(seed)
        Run modularity maximization
    calculate_zrand()
        Calculate zrand scores for each community assignment
    generate_mod_nulls(module_mappings, itr=25)
        Generate (hierarchical) modular null networks
    generate_MS_nulls(itr=10)
        Generate MS null networks
    """

    def __init__(self, config, w, net_data_path=None):

        super().__init__(config)

        self.w = w

        self.gamma = config.gamma
        self.nseeds = config.nseeds

        if net_data_path is not None:
            if os.path.exists(os.path.join(net_data_path, 'zrands.pickle')):
                with open(os.path.join(net_data_path, 'zrands.pickle'), 'rb') as f:
                    self.zrands = pickle.load(f)
                with open(os.path.join(net_data_path, 'max_zrand_idx.pickle'), 'rb') as f:
                    self.max_zrand_idx = pickle.load(f)
                with open(os.path.join(net_data_path, 'module_mappings.pickle'), 'rb') as f:
                    self.module_mappings = pickle.load(f)
                with open(os.path.join(net_data_path, 'module_mappings_level2.pickle'), 'rb') as f:
                    self.module_mappings_level2 = pickle.load(f)
            else:
                self.find_modules()
            if os.path.exists(os.path.join(net_data_path, 'strength.pickle')):
                with open(os.path.join(net_data_path, 'strength.pickle'), 'rb') as f:
                    self.strength = pickle.load(f)
            else:
                self.strength = np.sum(self.w, axis=0)
            if os.path.exists(os.path.join(net_data_path, 'mod_strength.pickle')):
                with open(os.path.join(net_data_path, 'mod_strength.pickle'), 'rb') as f:
                    self.mod_strength = pickle.load(f)
            else:
                self.mod_strength = {}
                for module in np.unique(self.module_mappings):
                    self.mod_strength[module] = np.mean(self.strength[self.module_mappings == module])
            if os.path.exists(os.path.join(net_data_path, 'clustering.pickle')):
                with open(os.path.join(net_data_path, 'clustering.pickle'), 'rb') as f:
                    self.clustering = pickle.load(f)
            else:
                self.clustering = bct.clustering_coef_wu(self.w)
            if os.path.exists(os.path.join(net_data_path, 'mod_clustering.pickle')):
                with open(os.path.join(net_data_path, 'mod_clustering.pickle'), 'rb') as f:
                    self.mod_clustering = pickle.load(f)
            else:
                self.mod_clustering = {}
                for module in np.unique(self.module_mappings):
                    self.mod_clustering[module] = np.mean(self.clustering[self.module_mappings == module])
            if os.path.exists(os.path.join(net_data_path, 'communicability.pickle')):
                with open(os.path.join(net_data_path, 'communicability.pickle'), 'rb') as f:
                    self.communicability = pickle.load(f)
            else:
                comm = communicability_wei(self.w)
                #Add NaNs to diagonal
                np.fill_diagonal(comm, np.nan)
                self.communicability = np.nanmean(comm, axis=0)
            if os.path.exists(os.path.join(net_data_path, 'mod_communicability.pickle')):
                with open(os.path.join(net_data_path, 'mod_communicability.pickle'), 'rb') as f:
                    self.mod_communicability = pickle.load(f)
            else:
                self.mod_communicability = {}
                for module in np.unique(self.module_mappings):
                    self.mod_communicability[module] = np.mean(self.communicability[self.module_mappings == module])
        else:
            self.find_modules()
            self.strength = np.sum(self.w, axis=0)
            self.mod_strength = {}
            for module in np.unique(self.module_mappings):
                self.mod_strength[module] = np.mean(self.strength[self.module_mappings == module])
            self.clustering = bct.clustering_coef_wu(self.w)
            self.mod_clustering = {}
            for module in np.unique(self.module_mappings):
                self.mod_clustering[module] = np.mean(self.clustering[self.module_mappings == module])
            comm = communicability_wei(self.w)
            #Add NaNs to diagonal
            np.fill_diagonal(comm, np.nan)
            self.communicability = np.nanmean(comm, axis=0)
            self.mod_communicability = {}
            for module in np.unique(self.module_mappings):
                self.mod_communicability[module] = np.mean(self.communicability[self.module_mappings == module])

        self.mod_nulls, self.mod_nulls_eff = self.generate_mod_nulls(np.array([self.module_mappings]))
        self.hierarchical_mod_nulls, self.hierarchical_mod_nulls_eff = self.generate_mod_nulls(np.array([self.module_mappings,
                                                                                                         self.module_mappings_level2]))
        self.ms_nulls, self.ms_nulls_eff = self.generate_MS_nulls()

        self.nets = {'empirical': [self.w],
                     'modular': self.mod_nulls,
                     'hierarchical modular': self.hierarchical_mod_nulls,
                     'MS': self.ms_nulls}

    def find_modules(self):

        self.cis, self.qs = list(zip(*(self.run_modularity(seed) for seed in range(self.nseeds))))
        self.zrands = np.sum(self.calculate_zrand(), axis=1)
        self.max_zrand_idx = np.argmax(self.zrands)

        self.module_mappings = self.cis[self.max_zrand_idx][0]
        #Check that the solution is hierarchical
        if self.cis[self.max_zrand_idx].shape[1] == 1:
            raise ValueError('The Louvain solution is not hierarchical.')
        self.module_mappings_level2 = self.cis[self.max_zrand_idx][1]

    def run_modularity(self, seed):
        if seed % 100 == 0:
            print(f'Running modularity maximization {seed} of {self.nseeds}')
        return bct.modularity_louvain_und(self.w, gamma=self.gamma, hierarchy=True, seed=seed)

    #Calculate zrand score for each community assignment across all other seeds
    def calculate_zrand(self):

        zrand_mat = np.zeros((self.nseeds, self.nseeds))
        for i in range(self.nseeds):
            if i % 100 == 0:
                print(f'Calculating zrand score {i} of {self.nseeds}')
            for j in range(i, self.nseeds):
                curr_zrand = zrand(self.cis[i][0], self.cis[j][0])
                zrand_mat[i, j] = curr_zrand
                zrand_mat[j, i] = curr_zrand

        return zrand_mat

    def generate_mod_nulls(self, module_mappings, itr=25):

        nulls = Parallel(n_jobs=self.njobs)(delayed(randmio_und_hmod)(self.w, module_mappings, itr=itr, seed=seed) for seed in range(self.nnetworks))
        nulls, nulls_eff = list(zip(*nulls))

        return nulls, nulls_eff

    def generate_MS_nulls(self, itr=10):

        nulls = Parallel(n_jobs=self.njobs)(delayed(bct.randmio_und_connected)(self.w, itr, seed=seed) for seed in range(self.nnetworks))
        nulls, nulls_eff = list(zip(*nulls))

        return nulls, nulls_eff