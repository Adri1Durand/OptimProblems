"""
Wrappers d'exécution des solveurs pour la campagne de benchmark :
    - GCO avec MADS comme sous-optimiseur, sélection de sous-espace PLS
    - MADS seul (PyNomad), utilisé comme référence

Chaque wrapper retourne un historique standardisé de la convergence pour
permettre la construction des data/performance profiles.
"""

import os
import sys
import numpy as np
import PyNomad
from GCO import GCO, mads_suboptimizer


from config import BUDGET_MAX_SIMPLEX_EVAL, DOE_SIZE, DOE_SEED, RESULTS_DIR



def run_gco_mads(problem, subspace_dimension, subspace_method="PLS", budget=BUDGET_MAX_SIMPLEX_EVAL, contraction=0.5,gamma=0.9, result_dir=RESULTS_DIR):
    """
    Lance GCO avec MADS comme sous-optimiseur et sélection de sous-espace PLS.

    Parameters
    ----------
    problem : dict
        Dictionnaire décrivant le problème (cf. problems.load_problems).
    subspace_dimension : int
        Dimension du sous-espace utilisé par la méthode PLS.
    run_seed : int
        Seed de l'exécution (indépendante du DOE).

    Returns
    -------
    dict : historique standardisé {"f_hist": np.ndarray, "n_eval": int}
    """
    n = problem["dimension"]
    lb = np.array(problem['bounds_low'], dtype=float)
    ub = np.array(problem['bounds_high'], dtype=float)
    problem_name = f"{problem['name']}_{n}D"

    var = {"dim": n, "lower_bound": lb, "upper_bound": ub}

    initial_doe = np.array(problem['initial_doe'], dtype=float)

    method_tag = f"GCO(MADS,{subspace_method},p{subspace_dimension},c{contraction},g{gamma})"
    file_path = os.path.join(
        result_dir, f"{method_tag}_{problem_name}.txt"
    )

    opt = {
        "file_path": file_path,
        "subspace_dimension": subspace_dimension,
        "subspace_selection": subspace_method,
        "min_termination_scale_parameter": 1e-9,
        "contraction" : contraction,
        "gamma":gamma,
    }

    fx_func = lambda x: problem["instance"].fx(x)

    solver = GCO(fx_func,vars_prop=var,opt=opt)

    # TODO: exécuter GCO -- confirmer le nom exact de la méthode de lancement
    X = initial_doe  # point de départ pour GCO (même DOE que pour MADS)
    F = np.array([fx_func(x) for x in X])
    solver.run_optim(X0=X,F0=F,suboptimizer = mads_suboptimizer, budget=n*budget)

    return 1 

def run_mads(problem, budget, result_dir=RESULTS_DIR):
    """
    Lance MADS seul (PyNomad) comme référence.

    Parameters
    ----------
    problem : dict
        Dictionnaire décrivant le problème (cf. problems.load_problems).
    run_seed : int
        Seed de l'exécution (indépendante du DOE).

    Returns
    -------
    dict : historique standardisé {"f_hist": np.ndarray, "n_eval": int}
    """
    n = problem["dimension"]
    lb = np.array(problem['bounds_low'], dtype=float)
    ub = np.array(problem['bounds_high'], dtype=float)
    problem_name = f"{problem['name']}_{n}D"

    def bb(eval_point):
        x = np.array([eval_point.get_coord(i) for i in range(eval_point.size())])
        f = problem["instance"].fx(np.array(x))  
        eval_point.setBBO(str(f).encode("UTF-8"))
        return 1 # 1: success 0: failed evaluation
    
    file_path = os.path.join(result_dir, f"MADS_{problem_name}.txt")

    params = [
        f"DIMENSION {n}",                     # <-- IMPORTANT, à ajouter explicitement
        "BB_OUTPUT_TYPE OBJ",
        "DISPLAY_DEGREE 0",
        "DISPLAY_ALL_EVAL false",
        "DISPLAY_STATS BBE OBJ MESH_SIZE POLL_SIZE",
        f"STATS_FILE {file_path} BBE OBJ",    # <-- STATS_FILE, pas STAT_FILE
        f"MAX_BB_EVAL {n * budget}",
        "MIN_FRAME_SIZE * 1e-9",              # <-- syntaxe correcte avec wildcard
    ]

    if n > 50 :
        params.append("DIRECTION_TYPE ortho 2n")
        params.append("NM_SEARCH false")


    X0 = np.array(problem['initial_doe'], dtype=float)
    # Écriture dans le fichier x0.txt
    with open("x0.txt", "w") as f:
        for p in X0:
            line = " ".join(f"{v:.6f}" for v in p)
            f.write(line + "\n")

    params.append("X0 x0.txt")

    result = PyNomad.optimize(bb,[],lb,ub,params)
    return 1 


