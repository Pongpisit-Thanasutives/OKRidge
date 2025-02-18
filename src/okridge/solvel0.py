import sys
import numpy as np
from tqdm import trange
from sklearn.preprocessing import normalize
from .tree import BNBTree

def okridge_solvel0(X, y, k=sys.maxsize, lambda2=1e-5, gap_tol=1e-4, norm=None, useBruteForce=True, time_limit=180, verbose=False):
    k = min(k, X.shape[-1])
    if norm is not None:
        X, norms = normalize(X, norm=norm, axis=0, return_norm=True)
    BnB_optimizer = BNBTree(X=X, y=y.flatten(), lambda2=lambda2, useBruteForce=useBruteForce, verbose=verbose)
    _, beta, _, _, _ = BnB_optimizer.solve(k=int(k), gap_tol=gap_tol, verbose=verbose, time_limit=time_limit)
    nonz_indices = np.nonzero(beta)[0].tolist()
    beta = beta/norms if norms is not None else beta
    return beta, nonz_indices

def okridge_solvel0_full(X, y, k=sys.maxsize, lambda2=1e-5, gap_tol=1e-4, norm=None, useBruteForce=True, time_limit=180, verbose=False):
    k = min(k, X.shape[-1])
    if norm is not None:
        X, norms = normalize(X, norm=norm, axis=0, return_norm=True)
    BnB_optimizer = BNBTree(X=X, y=y.flatten(), lambda2=lambda2, useBruteForce=useBruteForce, verbose=verbose)
    betas = np.zeros((k, X.shape[-1]))
    nonzero_indices = []
    for ss in trange(1, k+1):
        _, beta, _, _, _ = BnB_optimizer.solve(k=ss, gap_tol=gap_tol, verbose=verbose, time_limit=time_limit)
        nonz_indices = np.nonzero(beta)[0].tolist()
        betas[ss-1] = beta if norm is None else beta/norms
        nonzero_indices.append(nonz_indices)
    return betas, nonzero_indices

