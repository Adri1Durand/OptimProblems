import os
import sys
import argparse
import multiprocessing as mp

from config import BUDGET_MAX_SIMPLEX_EVAL, RESULTS_DIR
from solvers import run_gco_mads, run_mads


def runner_for_multiprocessing(problem, optimizer_name, subspace_dimension, subspace_method, ):

    n = problem["dimension"]
    budget = n * BUDGET_MAX_SIMPLEX_EVAL
    if subspace_dimension > n or subspace_dimension < 0:
        raise ValueError(f"Subspace dimension {subspace_dimension} is invalid for problem dimension {n}.")

    # Creating multiprocessing instance (object)
    ctx = mp.get_context("spawn") 
    # Spawning Non-Myopic isolated process
    q_nm = ctx.Queue()

    # ----------------
    # Select the optimizer and args
    # ----------------
    if subspace_dimension == 0:
        if optimizer_name == "MADS":
            optim_runner_process = run_mads
            args = (problem, budget, RESULTS_DIR)
        else:
            raise ValueError("Unsupported optimizer for subspace dimension 0")

    else:
        if optimizer_name == "MADS":
            optim_runner_process = run_gco_mads
            contraction = 0.5 
            gamma = 0.9
            
            args = (problem, subspace_dimension, subspace_method, budget, contraction, gamma, RESULTS_DIR)
        else:
            raise ValueError("Unsupported optimizer for subspace dimension > 0")

    # ----------------
    # Run the process
    # ----------------
    processus = ctx.Process(
        target=optim_runner_process, # fonction maths
        args=args # arguments de cette fonction maths
        ) # lancer un processus dédié pour ta ou tes fonctions

    processus.start() # lancer une des fonction maths 
    processus.join() # transmettre qu'on a fini d'exécuter le processus 


def main():
    parser = argparse.ArgumentParser(description="Run custom test for GCO settings.")
    parser.add_argument('--problem', type = str, required=True, 
                        help='Problem name and __dimension to run the test on. (ex., "ARGLINA__18")')
    parser.add_argument('--optimizer', type = str, default="MADS", choices=["TREGO","MADS"], 
                        help='Optimizer name to run the test on. (ex., "TREGO" or "MADS")')
    parser.add_argument('--subspace_dimension', type = int, default=0, 
                        help = "Dimension of the subspace framework GCO if p!=0 or not use the framework if p=0.")
    parser.add_argument('--subspace_method', type = str, default = "PLS", choices = ["PLS","PCA","None"],
                        help = "Method for subspace projection.")

    args = parser.parse_args()
    print("Test settings:"
          f"\nProblem: {args.problem}"
          f"\nOptimizer: {args.optimizer}"
          f"\nSubspace Dimension: {args.subspace_dimension}"
          f"\nSubspace Method: {args.subspace_method}")
    # To run the test, launch the script with the following command:
    # python TestSept26\\run_cutest_gco_settings.py --problem test1 --optimizer MADS --subspace_dimension 5 --subspace_method PLS


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    main()