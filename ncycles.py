import networkx as nx

def compute_ncycles(g):

    n_cycles = {n: 0 for n in range(1, 6)}
    for cycle in nx.simple_cycles(g, length_bound=5):
        n_cycles[len(cycle)] += 1

    return n_cycles