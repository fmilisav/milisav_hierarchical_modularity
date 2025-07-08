import numpy as np
from bct.utils import get_rng

#adapted from https://github.com/aestrivex/bctpy/blob/master/bct/algorithms/reference.py#L1479
#Hierarchical modularity and degree-preserving rewiring
def randmio_und_hmod(R, partitions, itr=10, seed=None):

    R = R.copy()
    if not np.allclose(R, R.T):
        raise ValueError("Input must be undirected")
    n = len(R)
    nlevels = partitions.shape[0]
    
    #Edge categories
    category_bool = np.zeros((nlevels + 1, n, n), dtype=bool)
    for level in range(nlevels):
        partition = partitions[level]
        for module in np.unique(partition):
            nodes = np.where(partition == module)[0]
            within_module_idx = np.ix_(nodes, nodes)
            category_bool[level][within_module_idx] = True
    
    category_bool[-1] = True

    #In reverse order of level, take out the indices of the previous level
    for level in range(nlevels, 0, -1):
        category_bool[level] = category_bool[level] & ~category_bool[level - 1]

    #Rewiring within each category
    B = np.zeros((n, n))
    eff_hierarchy = []
    for level in range(nlevels + 1):
        R_category = R.copy()
        R_category[~category_bool[level]] = 0
        B_category, eff = category_rewiring(R_category, category_bool[level], 
                                            itr, seed)
        B += B_category
        eff_hierarchy.append(eff)

    return B, eff_hierarchy

#Within-category degree-preserving rewiring
def category_rewiring(R, category_bool, itr=10, seed=None):

    R = R.copy()
    if not np.allclose(R, R.T):
        raise ValueError("Input must be undirected")

    rng = get_rng(seed)
    n = len(R)
    i, j = np.where(np.tril(R))
    k = len(i)
    itr *= k

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1, e2 = rng.randint(k, size=(2,))
                while e1 == e2:
                    e2 = rng.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            if rng.random_sample() > .5:
                i.setflags(write=True)
                j.setflags(write=True)
                i[e2] = d
                j[e2] = c  # flip edge c-d with 50% probability
                c = i[e2]
                d = j[e2]  # to explore all potential rewirings

            # rewiring condition
            # making sure the new edges remain in the same category
            if category_bool[a, d] and category_bool[c, b]:
                if not (R[a, d] or R[c, b]):
                    R[a, d] = R[a, b]
                    R[a, b] = 0
                    R[d, a] = R[b, a]
                    R[b, a] = 0
                    R[c, b] = R[c, d]
                    R[c, d] = 0
                    R[b, c] = R[d, c]
                    R[d, c] = 0

                    j.setflags(write=True)
                    j[e1] = d
                    j[e2] = b  # reassign edge indices
                    eff += 1
                    break
            att += 1

    return R, eff