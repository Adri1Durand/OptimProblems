"""
onejob_runner.py

Exécute un run d'optimisation (MADS ou GCO-MADS) sur UN SEUL problème S2MPJ.
Destiné à être lancé indépendamment, problème par problème, sur une grille
de calcul (un job = un problème).

Usage:
    Go to the directory containing this script and run the following command in the terminal:
    >> cd TestSept26
    Create a results directory if it doesn't exist:
    >> mkdir results
    Launch one job for a specific problem (e.g., ARGLINA) with MADS and a subspace dimension of 5 using PLS:
    >>python onejob_runner.py --problem ARGLINA --optimizer MADS --subspace_dimension 5 --subspace_method PLS
"""

import os
import json
import argparse
import multiprocessing as mp
import numpy as np

from config import BUDGET_MAX_SIMPLEX_EVAL, RESULTS_DIR
from solvers import run_gco_mads, run_mads
from s2mpj_loader import setup_s2mpj_path, get_s2mpj_class

def load_problem_instance(problem_name, json_path=None):
    """
    Charge une instance de problème S2MPJ à partir de son nom,
    et récupère les métadonnées (bornes, x0, DOE...) depuis le JSON.

    Parameters
    ----------
    problem_name : str
        Nom du problème S2MPJ (ex: "ARGLINA").
    json_path : str, optional
        Chemin vers le fichier JSON contenant les métadonnées des problèmes.
        Si None, cherche "problems_list.json" dans le même dossier que ce fichier.

    Returns
    -------
    dict
        Dictionnaire complet du problème, prêt à être utilisé par les solvers.
    """
    setup_s2mpj_path()

    if json_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "problems_list.json")

    with open(json_path, "r") as f:
        problems_meta = json.load(f)

    # Cherche l'entrée correspondant au nom du problème
    meta = next((p for p in problems_meta if p["name"] == problem_name), None)
    if meta is None:
        raise ValueError(
            f"Aucune métadonnée trouvée pour le problème '{problem_name}' dans {json_path}"
        )

    param = meta["param"]
    pb_class = get_s2mpj_class(problem_name)

    # Instanciation avec le bon paramètre de dimension (comme load_problems_from_json)
    try:
        if isinstance(param, list):
            instance = pb_class(*param)
        else:
            instance = pb_class(param)
    except Exception as e:
        raise RuntimeError(
            f"Erreur lors de l'instanciation de '{problem_name}' avec param={param}: {e}"
        )

    problem = {
        "name": meta["name"],
        "dimension": meta["dimension"],
        "instance": instance,
        "x0": np.array(meta["x0"]),
        "bounds_low": np.array(meta["bounds_low"]),
        "bounds_high": np.array(meta["bounds_high"]),
        "initial_doe": np.array(meta["initial_doe"]) if meta.get("initial_doe") else None,
        "doe_size": meta.get("doe_size"),
        "shift": np.array(meta["shift"]) if meta.get("shift") else None,
    }

    return problem

def run_problem_process(problem, optimizer_name, subspace_dimension, subspace_method):
    """
    Lance l'optimisation sur `problem` dans un process séparé (isolation mémoire/crash).
    """
    n = problem["dimension"]
    budget = n * BUDGET_MAX_SIMPLEX_EVAL

    if subspace_dimension > n or subspace_dimension < 0:
        raise ValueError(
            f"Subspace dimension {subspace_dimension} is invalid for problem dimension {n}."
        )

    ctx = mp.get_context("spawn")

    if subspace_dimension == 0:
        if optimizer_name == "MADS":
            target_fn = run_mads
            proc_args = (problem, budget, RESULTS_DIR)
        else:
            raise ValueError("Unsupported optimizer for subspace dimension 0")
    else:
        if optimizer_name == "MADS":
            target_fn = run_gco_mads
            contraction = 0.5
            gamma = 0.9
            proc_args = (
                problem, subspace_dimension, subspace_method,
                budget, contraction, gamma, RESULTS_DIR,
            )
        else:
            raise ValueError("Unsupported optimizer for subspace dimension > 0")

    process = ctx.Process(target=target_fn, args=proc_args)
    process.start()
    process.join()


def main():
    parser = argparse.ArgumentParser(
        description="Run a single S2MPJ problem with MADS/GCO-MADS (one job = one problem)."
    )
    parser.add_argument(
        "--problem", type=str, required=True,
        help='Nom du problème S2MPJ (ex., "ARGLINA")',
    )
    parser.add_argument(
        "--optimizer", type=str, default="MADS", choices=["TREGO", "MADS"],
        help='Optimiseur à utiliser.',
    )
    parser.add_argument(
        "--subspace_dimension", type=int, default=0,
        help="Dimension du sous-espace GCO (0 = pas de GCO).",
    )
    parser.add_argument(
        "--subspace_method", type=str, default="PLS", choices=["PLS", "PCA", "None"],
        help="Méthode de projection du sous-espace.",
    )

    args = parser.parse_args()
    print(
        "Test settings:"
        f"\nProblem: {args.problem}"
        f"\nOptimizer: {args.optimizer}"
        f"\nSubspace Dimension: {args.subspace_dimension}"
        f"\nSubspace Method: {args.subspace_method}"
    )

    problem = load_problem_instance(args.problem)


    run_problem_process(
        problem, args.optimizer, args.subspace_dimension, args.subspace_method
    )


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    main()