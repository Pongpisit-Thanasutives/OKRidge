import numpy as np
from .tree import BNBTree

def okridge_solvel0(X, y, k, lambda2=1e-5, gap_tol=1e-6, time_limit=300, verbose=False):
    BnB_optimizer = BNBTree(X=X, y=y.flatten(), lambda2=lambda2, verbose=verbose)
    _, beta, _, _, _ = BnB_optimizer.solve(k=int(k), gap_tol=gap_tol, verbose=verbose, time_limit=time_limit)
    return beta, np.nonzero(beta)[0].tolist()

def okridge_solvel0_full(X, y, k, lambda2=1e-5, gap_tol=1e-6, time_limit=300, verbose=False):
    BnB_optimizer = BNBTree(X=X, y=y.flatten(), lambda2=lambda2, verbose=verbose)
    betas = np.zeros((k, X.shape[-1]))
    nonzero_indices = []
    for ss in range(1, k+1):
        _, beta, _, _, _ = BnB_optimizer.solve(k=int(ss), gap_tol=gap_tol, verbose=verbose, time_limit=time_limit)
        nonz_indices = np.nonzero(beta)[0].tolist()
        betas[k-1] = beta
        nonzero_indices.append(nonz_indices)
    return betas, nonzero_indices

