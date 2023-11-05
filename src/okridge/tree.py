import time
import queue
import sys
from collections import namedtuple

import numpy as np

from .node import Node, branch  # , presolve


from .solvers import (
    sparseLogRegModel_big_n as sparseLogRegModel_big_n_with_cache,
    total_size,
)

import psutil
import gc
import scipy

from .utils import (
    get_RAM_available_in_bytes,
    get_RAM_available_in_GB,
    get_RAM_used_in_bytes,
    get_RAM_used_in_GB,
)


class DataClass:
    def __init__(self, X, y, lambda2):
        """Create a DataClass object used to store data for the BnBTree class

        Args:
            X (np.array): n x p numpy array of features
            y (np.array): 1 dimensional numpy array of size n of predictions
            lambda2 (float): coefficient of l2 regularization
        """
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y
        self.X_norm_2 = np.linalg.norm(X, axis=0) ** 2
        self.XTX = X.T @ X
        # self.X_norm_2 = self.XTX.diagonal() ** 2
        self.XTX_lambda2 = self.XTX + lambda2 * np.eye(N=self.p)
        self.XTy = y.dot(self.X)
        self.lambda2 = lambda2
        self.half_Lipschitz = self.XTX_lambda2.diagonal()

        eigenvalues = np.linalg.eigvalsh(self.XTX)
        self.smallest_eigenval = np.linalg.eigvalsh(self.XTX)[0] * 0.999
        print("we take lambda to be", self.smallest_eigenval)

        self.rho_ADMM = 2 / np.sqrt(
            (eigenvalues[1] - self.smallest_eigenval)
            * (eigenvalues[-1] - self.smallest_eigenval)
        )

        self.rho_ADMM_is_finetuned = True
        if eigenvalues[1] < self.smallest_eigenval + 1e-6:
            self.rho_ADMM_is_finetuned = False

        if self.smallest_eigenval < 1e-3:
            self.smallest_eigenval = 0.0
        self.XTX_minus_smallest_eigenval = (
            self.XTX - np.eye(self.p) * self.smallest_eigenval
        )
        self.lambda2_plus_smallest_eigenval = self.lambda2 + self.smallest_eigenval


class BNBTree:
    def __init__(
        self,
        X,
        y,
        int_tol=1e-4,
        gap_tol=1e-4,
        lambda2=1e-5,
        max_memory_GB=300,
        useBruteForce=False,
        tighten_bound_via_ADMM=True,
    ):
        """Initialize the BnBTree class with the data and parameters

        Args:
            X (np.array): n x p numpy array of features
            y (np.array): 1 dimensional numpy array of size n of predictions
            int_tol (float, optional): integer tolerance hyperparameter. Defaults to 1e-4.
            gap_tol (float, optional): optimality gap tolerance hyperparameter. Defaults to 1e-4.
            lambda2 (float, optional): coefficient of l2 regularization. Defaults to 1e-5.
            max_memory_GB (int, optional): max memory to use to store all unprocessed nodes and heuristic solutions. Defaults to 300.
            useBruteForce (bool, optional): use brute force search when the number of enumeration is small. Defaults to False.
            tighten_bound_via_ADMM (bool, optional): whether to use ADMM to tighten the lower bound computed by the Fast Solve method. Defaults to True.
        """
        data = DataClass(X, y, lambda2)
        self.data = data

        self.int_tol = int_tol
        self.gap_tol = gap_tol

        self.useBruteForce = useBruteForce
        self.bruteForceThreshold = 1000

        self.tighten_bound_via_ADMM = tighten_bound_via_ADMM

        self.bfs_queue = None
        self.dfs_queue = None

        self.levels = {}
        self.number_of_nodes = 0

        self.root = None

        available_memory_GB = get_RAM_available_in_GB()
        if max_memory_GB is None:
            print(
                "No max_memory_GB is given. Using all available memory ({} GB) in the machine".format(
                    available_memory_GB
                )
            )
            self.max_memory_GB = available_memory_GB
        elif max_memory_GB > available_memory_GB:
            print(
                "max_memory_GB is larger than available memory. Using all available memory ({} GB) in the machine".format(
                    available_memory_GB
                )
            )
            self.max_memory_GB = available_memory_GB
        else:
            print("Using max memory ({} GB)".format(max_memory_GB))
            self.max_memory_GB = max_memory_GB
        self.safe_max_memory_GB = 0.95 * self.max_memory_GB
        self.RAM_used_GB_start = get_RAM_used_in_GB()

        self.upper_solver_with_cache = sparseLogRegModel_big_n_with_cache(
            data=self.data,
            intercept=False,
            parent_size=50,
            child_size=50,
            max_memory_GB=50,
        )

    def get_RAM_used_since_start(self):
        RAM_used_GB = get_RAM_used_in_GB()
        return RAM_used_GB - self.RAM_used_GB_start

    def solve(
        self,
        k,
        gap_tol=1e-2,
        number_of_dfs_levels=0,
        verbose=False,
        time_limit=3600,
    ):
        """Solve the k-sparse ridge regression problem using branch and bound

        Args:
            k (int): cardinality constraint
            gap_tol (float, optional): optimality gap tolerance hyperparameter. Defaults to 1e-2.
            number_of_dfs_levels (int, optional): number of levels for depth-first search during branch and bound. Defaults to 0.
            verbose (bool, optional): whether to print informations into terminal. Defaults to False.
            time_limit (int, optional): time limit (in seconds) of running branch and bound. Defaults to 3600.

        Returns:
            float: cost or loss of the best solution found
            np.array: best solution found
            float: running time of branch and bound
            float: best lower bound found
            float: best gap found
        """

        st = time.time()
        upper_beta = np.zeros((self.data.p,))
        upper_bound = self.data.XTX_lambda2.dot(upper_beta).dot(upper_beta)

        if verbose:
            print(f"initializing took {time.time() - st} seconds")

        st = time.time()
        # root node
        self.root = Node(None, None, None, data=self.data)

        self.bfs_queue = queue.Queue()
        self.dfs_queue = queue.LifoQueue()
        self.bfs_queue.put(self.root)

        # lower and upper bounds initialization
        dual_bound = {}
        self.levels = {0: 1}
        min_open_level = 0

        max_lower_bound_value = -sys.maxsize
        best_gap = gap_tol + 1

        if verbose:
            print(f"{number_of_dfs_levels} levels of depth used")

        upper_bound = self.root.upper_solve_with_cache(k, self.upper_solver_with_cache)
        upper_beta = self.root.upper_beta.copy()
        if (k == 1) or (time_limit < 0):
            max_lower_bound_value = upper_bound
            best_gap = 0
            return (
                upper_bound,
                upper_beta,
                best_gap,
                max_lower_bound_value,
                time.time() - st,
            )
        
        # print("initial upper_bound is", upper_bound)
        # print("initial nonzero indices are", np.nonzero(upper_beta)[0])

        if self.root.data.rho_ADMM_is_finetuned is False:
            self.root.finetune_ADMM_rho(k, upper_bound, factor=10)

        RAM_used_GB_since_start = self.get_RAM_used_since_start()
        # keep searching through the queue if the queue is not empty AND time limit is not reached
        while (
            (self.bfs_queue.qsize() > 0 or self.dfs_queue.qsize() > 0)
            and (time.time() - st < time_limit)
            and (RAM_used_GB_since_start < self.safe_max_memory_GB)
        ):

            gc.collect()

            RAM_used_GB_since_start = self.get_RAM_used_since_start()

            # get current node
            if self.dfs_queue.qsize() > 0:
                curr_node = self.dfs_queue.get()
            else:
                curr_node = self.bfs_queue.get()

            # prune?
            if (
                curr_node.parent_lower_bound
                and upper_bound <= curr_node.parent_lower_bound
            ):
                self.levels[curr_node.level] -= 1
                continue

            # calculate dual values
            curr_dual = self._solve_node(
                curr_node, k, dual_bound, upper_bound, tighten_bound_via_ADMM=self.tighten_bound_via_ADMM
            )

            # prune?with newly calculated lower bound
            if curr_dual >= upper_bound:
                curr_node.delete_storedData_on_allowed_support()
                continue

            curr_upper_bound = curr_node.upper_solve_with_cache(
                k, self.upper_solver_with_cache
            )

            # if verbose:
            #     print("curr_upper_bound: {}, curr_dual: {}".format(curr_upper_bound, curr_dual)) # debugging

            if curr_upper_bound < upper_bound:
                upper_bound = curr_upper_bound
                upper_beta = curr_node.upper_beta.copy()
                best_gap = (upper_bound - max_lower_bound_value) / abs(upper_bound)
                # print("********************************\nfind better solution with upper_bound", upper_bound)
                # print("the nonzero indices are", np.nonzero(upper_beta)[0], "\n********************************")

            # update gap?
            if self.levels[min_open_level] == 0:
                print(
                    "there are {} nodes left in the bfs queue".format(
                        self.bfs_queue.qsize()
                    )
                )
                del self.levels[min_open_level]
                max_lower_bound_value = max(
                    [j for i, j in dual_bound.items() if i <= min_open_level]
                )
                best_gap = (upper_bound - max_lower_bound_value) / abs(upper_bound)
                if verbose:
                    print(
                        f"l: {min_open_level}, (d: {max_lower_bound_value}, "
                        f"u: {upper_bound}, g: {best_gap}, "
                        f"t: {time.time() - st} s"
                    )
                min_open_level += 1

            # arrived at a solution?
            if best_gap <= gap_tol:
                # print("reaching gap_tol mid way!")
                # print(
                #     "there are {} nodes left".format(
                #         self.bfs_queue.qsize() + self.dfs_queue.qsize()
                #     )
                # )
                return (
                    upper_bound,
                    upper_beta,
                    best_gap,
                    max_lower_bound_value,
                    time.time() - st,
                )

            # branch?
            curr_gap = (curr_upper_bound - curr_dual) / abs(curr_upper_bound)
            if curr_gap <= gap_tol:
                # print("curr_gap is smaller than gap_tol; skipping!")
                pass
            elif (curr_dual < upper_bound) and (
                len(curr_node.fixed_support_on_allowed_support) < k
            ):
                left_node, right_node = branch(curr_node, k)
                self.levels[curr_node.level + 1] = (
                    self.levels.get(curr_node.level + 1, 0) + 2
                )
                if curr_node.level < min_open_level + number_of_dfs_levels:
                    self.dfs_queue.put(right_node)
                    self.dfs_queue.put(left_node)
                else:
                    self.bfs_queue.put(right_node)
                    self.bfs_queue.put(left_node)
            else:
                # print(
                #     "fixed support size is {}".format(
                #         len(curr_node.fixed_support_on_allowed_support)
                #     )
                # )
                # print("no branching!!!")
                pass

            curr_node.delete_storedData_on_allowed_support()
            # del curr_node

            # print("total size of saved_solution is", total_size(self.upper_solver_with_cache.saved_solution))
        
        # print("counting number of heuristic solutions")
        # print("number of heuristic solutions is", len(self.upper_solver_with_cache.saved_solution))
        # losses = []
        # sub_betas = []
        # indices_strs = []
        # yTy = self.data.y.dot(self.data.y)

        # for indices_str in self.upper_solver_with_cache.saved_solution.keys():
        #     sub_beta, _, loss = self.upper_solver_with_cache.saved_solution[indices_str]
        #     losses.append(loss + yTy)
        #     sub_betas.append(sub_beta)
        #     indices_strs.append(indices_str)
        
        # loss_indices = np.argsort(losses)
        # print("smallest 10 losses are", np.sort(losses)[:30])

        # beta_collections = np.zeros((1000, self.data.p))
        # loss_collections = []
        # for i in range(1000):
        #     loss_collections.append(losses[loss_indices[i]])
        #     nonzero_indices = np.fromstring(indices_strs[loss_indices[i]], dtype=bool).nonzero()[0].astype(int)
        #     print("nonzero_indices are", nonzero_indices)
        #     print("sub_beta is", sub_betas[loss_indices[i]])
        #     beta_collections[i, nonzero_indices] = sub_betas[loss_indices[i]]

        if not (self.bfs_queue.qsize() > 0 or self.dfs_queue.qsize() > 0):
            # print("There are no nodes left in the queue")
            # update lower bound and gap
            max_lower_bound_value = upper_bound
            best_gap = 0.0
        elif RAM_used_GB_since_start >= self.safe_max_memory_GB:
            print(
                "RAM used since start is greater than 0.95*max_memory(0.95*{} GB = {} GB)!".format(
                    self.max_memory_GB, self.safe_max_memory_GB
                )
            )
        else:
            print("Time limit is reached!")
        return (
            upper_bound,
            upper_beta,
            best_gap,
            max_lower_bound_value,
            time.time() - st,
        )

    def _solve_node(self, curr_node, k, dual_, upper_bound, tighten_bound_via_ADMM=True):
        """Solve the perspective relaxation of the node and update the dual bound

        Args:
            curr_node (CustomClass): current node to be solved
            k (int): cardinality constraint
            dual_ (list): list of minimum dual values at each depth of the tree
            upper_bound (float): current loss of the best solution found for this node
            tighten_bound_via_ADMM (bool, optional): whether to refine the dual bound computed by the Fast Solve method. Defaults to True.

        Returns:
            float: dual value of the perspective relaxation of the current node
        """
        self.number_of_nodes += 1
        total_num_bruteForce = scipy.special.comb(
            len(curr_node.unfixed_support_on_allowed_support),
            k - len(curr_node.fixed_support_on_allowed_support),
        )
        if self.useBruteForce and (total_num_bruteForce < self.bruteForceThreshold):
            curr_dual = curr_node.lower_solve_brute_force(
                k, self.upper_solver_with_cache
            )
            # print("lower bound by brute force method is", curr_dual)
        else:
            curr_dual = curr_node.lower_solve(k, upper_bound)
            # print("lower bound by fast method is", curr_dual)

            if tighten_bound_via_ADMM and (curr_dual < upper_bound):
                # curr_dual = curr_node.lower_solve_improve(k, upper_bound)
                curr_dual = curr_node.lower_solve_admm(k, upper_bound)
                # print("lower bound by admm method is", curr_dual)

        dual_[curr_node.level] = min(curr_dual, dual_.get(curr_node.level, sys.maxsize))
        self.levels[curr_node.level] -= 1
        return curr_dual

    def solve_root(
        self,
        k,
        gap_tol=1e-2,
        verbose=False,
    ):
        # should I keep this function?

        st = time.time()
        upper_beta = np.zeros((self.data.p,))
        upper_bound = self.data.XTX_lambda2.dot(upper_beta).dot(upper_beta)

        if verbose:
            print(f"initializing took {time.time() - st} seconds")

        st = time.time()
        # root node
        self.root = Node(None, None, None, data=self.data)

        dual_bound = {}
        self.levels = {0: 1}
        max_lower_bound_value = -sys.maxsize
        best_gap = gap_tol + 1

        # calculate upper bound
        upper_bound = self.root.upper_solve_with_cache(k, self.upper_solver_with_cache)
        upper_beta = self.root.upper_beta.copy()

        # calculate lower bound
        curr_dual = self._solve_node(
            self.root, k, dual_bound, upper_bound, tighten_bound_via_ADMM=self.tighten_bound_via_ADMM
        )
        max_lower_bound_value = curr_dual
        best_gap = (upper_bound - max_lower_bound_value) / abs(upper_bound)

        if k == 1:
            max_lower_bound_value = upper_bound
            best_gap = 0

        return (
            upper_bound,
            upper_beta,
            best_gap,
            max_lower_bound_value,
            time.time() - st,
        )