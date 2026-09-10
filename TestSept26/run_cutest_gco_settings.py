import os
import sys
import argparse
import multiprocessing


def main():
    parser = argparse.ArgumentParser(description="Run custom test for GCO settings.")
    parser.add_argument('--problem', type = str, required=True, 
                        help='Problem name and __dimension to run the test on. (ex., "ARGLINA__18")')
    parser.add_argument('--optimizer', type = str, default="MADS", choices=["TREGO","MADS"], 
                        help='Optimizer name to run the test on. (ex., "TREGO" or "MADS")')
    parser.add_argument('--subspace_dimension', type = int, default=0, 
                        help = "Dimension of the subspace framework GCO if p!=0 or not use the framework if p=0.")
    
    args = parser.parse_args()
    print("Test settings:"
          f"\nProblem: {args.problem}"
          f"\nOptimizer: {args.optimizer}"
          f"\nSubspace Dimension: {args.subspace_dimension}")
    # To run the test, launch the script with the following command:
    # python run_cutest_gco_settings.py --problem <problem_name> --optimizer <optimizer_name> --subspace_framework <True/False>
    # python TestSept26\\run_cutest_gco_settings.py --problem test1 --optimizer MADS --subspace_dimension 5


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    main()