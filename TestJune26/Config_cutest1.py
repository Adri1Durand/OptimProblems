import os
import sys
import inspect

import numpy as np

S2MPJ_ROOT = "/home/duraadri/Bureau/EssaisPython"
main_dir = os.path.join(S2MPJ_ROOT, "S2MPJ")
dir2 = os.path.join(main_dir, "python_problems")

for path in (main_dir, dir2):
    if path not in sys.path:
        sys.path.insert(0, path)

from ARGLINA import ARGLINA
from CHEBYQAD import CHEBYQAD
from EXPLIN import EXPLIN
from EXPLIN2 import EXPLIN2
from COSINE import COSINE
from HADAMARD import HADAMARD
from BDQRTIC import BDQRTIC
from LINVERSE import LINVERSE
from MCCORMCK import MCCORMCK
from NCVXBQP1 import NCVXBQP1
from NCVXBQP2 import NCVXBQP2
from NCVXBQP3 import NCVXBQP3
from NONSCOMP import NONSCOMP
from PENALTY2 import PENALTY2
from SINEALI import SINEALI
from CVXQP1 import CVXQP1
from NCVXQP1 import NCVXQP1
from NCVXQP2 import NCVXQP2
from NCVXQP3 import NCVXQP3
from S368 import S368
from NCVXQP4 import NCVXQP4
from BIGGSB1 import BIGGSB1
from EXTROSNB import EXTROSNB
# from TORSION1 import TORSION1
from POWELLSG import POWELLSG
from EXPQUAD import EXPQUAD
#from HARTMAN3 import HARTMAN3
# from MROSE import MROSE
# from QRTQUAD import QRTQUAD

list_classes = [
    ARGLINA, CHEBYQAD, EXPLIN, EXPLIN2, COSINE, HADAMARD,
    BDQRTIC, LINVERSE, MCCORMCK, NCVXBQP1, NCVXBQP2, NCVXBQP3,
    NONSCOMP, PENALTY2, SINEALI, CVXQP1, NCVXQP1,
    NCVXQP2, NCVXQP3, S368, NCVXQP4, BIGGSB1, EXTROSNB,
    POWELLSG, EXPQUAD
]

def inspect_dimension_formula(pb_class, params=[2, 3, 5, 7, 10, 15]):
    name = pb_class.__name__
    print(f"\n{name}:")
    for p in params:
        try:
            pb = pb_class(p)
            dim = len(np.asarray(pb.x0, dtype=float).flatten())
            print(f"  param={p:3d} â†’ len(x0)={dim:5d}  (dim-p={dim-p}, dim/p={dim/p:.2f})")
        except Exception as e:
            print(f"  param={p:3d} â†’ ERREUR: {e}")

# ============================================================
# DICTIONNAIRE : dimension cible â†’ PARAMÃˆTRE du constructeur
# (formules inverses dÃ©couvertes par inspection)
# ============================================================
PARAM_FROM_DIM = {
    # HADAMARD : dim = pÂ² + 1  â†’  p = sqrt(dim - 1)
    'HADAMARD': lambda d: max(int(np.round(np.sqrt(max(d - 1, 1)))), 2),

    # TORSION : dim = 4pÂ²  â†’  p = sqrt(dim)/2
    'TORSION1': lambda d: max(int(np.round(np.sqrt(d) / 2)), 2),
    'TORSION2': lambda d: max(int(np.round(np.sqrt(d) / 2)), 2),

    # CBRATU3D : dim = 2pÂ³  â†’  p = cbrt(dim/2)
    # CBRATU3D : dim = 2pÂ³, seul p=2 (dim=16) tient dans [10,49]
    'CBRATU3D': lambda d: 2,

    # BRATU1D : dimension paire  â†’  ajuster au pair le plus proche
    'BRATU1D': lambda d: d if d % 2 == 0 else d + 1,

    # BDQRTIC : n >= 5
    'BDQRTIC': lambda d: max(d, 5),

    # LINVERSE : dim = 2p - 1  â†’  p = (dim + 1) / 2   (p >= 3)
    'LINVERSE': lambda d: max(int(np.round((d + 1) / 2)), 3),

    # CATENARY : dim = 3p + 3  â†’  p = (dim - 3) / 3   (p >= 2)
    'CATENARY': lambda d: max(int(np.round((d - 3) / 3)), 2),

    # CYCLOOCT : dim = 3p  â†’  p = dim / 3   (p >= 3)
    'CYCLOOCT': lambda d: max(int(np.round(d / 3)), 3),
        
    # POWELLSG : param = N (dimension)
    'POWELLSG': lambda d: max(4 * int(np.round(d / 4)), 4),
}

# Par dÃ©faut : le paramÃ¨tre EST la dimension
DEFAULT_PARAM = lambda d: max(int(np.round(d)), 2)

def get_param_for_dim(name, target_dim):
    """Retourne le paramÃ¨tre Ã  passer au constructeur pour viser target_dim."""
    formula = PARAM_FROM_DIM.get(name, DEFAULT_PARAM)
    return formula(target_dim)

def _sample_shift(bounds_low, bounds_high, rng=None):
    """
    Tire alÃ©atoirement un point dans les bounds.
    Ce point servira de dÃ©calage pour la fonction.
    """
    rng = rng or np.random
    return rng.uniform(low=bounds_low, high=bounds_high)

# def _clip_to_bounds(x, xl, xu):
#     x = np.asarray(x, dtype=float).flatten()
#     xl = np.asarray(xl, dtype=float).flatten()
#     xu = np.asarray(xu, dtype=float).flatten()
#     return np.clip(x, xl, xu)

def _clip_to_bounds(x0, bounds_low, bounds_high, eps=1e-6):
    return np.clip(x0, bounds_low + eps, bounds_high - eps)

def generate_problem_instances(dim_range=(10, 49), doe_ratio=2, seed=None, versions=2):
    """
    GÃ©nÃ¨re `versions` instances de chaque problÃ¨me avec des dimensions diffÃ©rentes.
    
    Parameters:
    -----------
    dim_range : tuple (min_dim, max_dim)
        Plage de dimensions Ã  tirer alÃ©atoirement
    doe_ratio : int
        Ratio de points DOE = n * doe_ratio
    seed : int
        Graine alÃ©atoire
    versions : int
        Nombre de versions (dimensions diffÃ©rentes) par problÃ¨me
    """
    problems = []
    rng = np.random.default_rng(seed)

    for pb_class in list_classes:
        name = pb_class.__name__
        
        # GÃ©nÃ©rer `versions` instances avec dimensions diffÃ©rentes
        for version in range(1, versions + 1):
            try:
                # 1. Dimension cible demandÃ©e
                n_requested = rng.integers(*dim_range)
                while name == "S368" and n_requested > 367 :
                    n_requested = rng.integers(*dim_range)

                # 2. Calcul DIRECT du paramÃ¨tre
                param = get_param_for_dim(name, n_requested)

                # 3. UNE SEULE instanciation
                pb = pb_class(param)
                x0 = np.asarray(pb.x0, dtype=float).flatten()
                n_actual = len(x0)

                # 4. Bornes via xlower/xupper
                INF = 1e19
                if hasattr(pb, 'xlower') and pb.xlower is not None:
                    bounds_low = np.asarray(pb.xlower, dtype=float).flatten()
                else:
                    bounds_low = np.full(n_actual, -100.0)

                if hasattr(pb, 'xupper') and pb.xupper is not None:
                    bounds_high = np.asarray(pb.xupper, dtype=float).flatten()
                else:
                    bounds_high = np.full(n_actual, 100.0)

                if len(bounds_low) != n_actual:
                    bounds_low = np.full(n_actual, -100.0)
                if len(bounds_high) != n_actual:
                    bounds_high = np.full(n_actual, 100.0)

                bounds_low[bounds_low < -INF] = -100.0
                bounds_high[bounds_high > INF] = 100.0

                # 4bis. SHIFT : tirage alÃ©atoire d'un point dans les bounds
                shift = _sample_shift(bounds_low, bounds_high, rng=rng)

                # 5. DOE (construit autour de x0 original, dÃ©calÃ© du shift)
                doe_size = max(int(n_actual * doe_ratio) - 1, 1)
                lhs = rng.uniform(low=bounds_low, high=bounds_high,
                                  size=(doe_size, n_actual))
                
                # # --- Clip du x0 dans les bornes avant de l'ajouter au DOE ---
                # x0_clipped = _clip_to_bounds(x0, bounds_low, bounds_high)
                # initial_doe = np.vstack([lhs, x0_clipped])

                x0_clipped = _clip_to_bounds(x0, bounds_low, bounds_high, eps=1e-6)
                lhs = rng.uniform(low=bounds_low + 1e-6, high=bounds_high - 1e-6, size=(doe_size, n_actual))

                initial_doe = np.vstack([lhs, x0_clipped])

                problem_info = {
                    'name': name,
                    'version': version,
                    'class': pb_class,
                    'instance': pb,
                    'param': param,
                    'dimension': n_actual,
                    'dimension_requested': n_requested,
                    'x0': x0,
                    'shift': shift,
                    'bounds_low': bounds_low,
                    'bounds_high': bounds_high,
                    'initial_doe': initial_doe,
                    'doe_size': doe_size
                }
                problems.append(problem_info)

                delta = n_actual - n_requested
                shift_norm = np.linalg.norm(shift)
                bounds_str = f"[{bounds_low[0]:8.2f}, {bounds_high[0]:8.2f}]"
                print(f"âœ“ {name:15s} v{version} | dim_req={n_requested:4d} â†’ param={param:3d} "
                      f"â†’ dim={n_actual:4d} (Î”={delta:+d}) | doe={doe_size:4d} "
                      f"| bounds={bounds_str} | shift_norm={shift_norm:.4f}")

            except Exception as e:
                print(f"âœ— {name:15s} v{version} | Error: {str(e)}")
                continue

    print(f"\nâœ“ {len(problems)}/{len(list_classes) * versions} instances gÃ©nÃ©rÃ©es "
          f"({len(list_classes)} problÃ¨mes Ã— {versions} versions)")
    return problems

if __name__ == "__main__":
    problems = generate_problem_instances(dim_range=(50, 499), doe_ratio=2, versions=2)
    # Sauvegarder la liste dans un json pour usage futur
    import json

    def json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)  # pour les classes, instances, etc.

    with open("problems_mid_list.json", "w") as f:
        json.dump(problems, f, indent=2, default=json_default)

# RÃ©cupÃ©rer le problÃ¨me numÃ©ro 5
for problem in problems:
    # Afficher les infos
    print(f"ProblÃ¨me: {problem['name']} (version {problem['version']})")
    print(f"Dimension: {problem['dimension']}")

    # RÃ©cupÃ©rer l'instance du problÃ¨me
    pb = problem['instance']

    # âœ… Utiliser les bonnes mÃ©thodes CUTEst
    x0 = np.asarray(problem['x0']).flatten()

    # Ã‰valuation simple (juste la fonction)
    f_x0 = pb.fx(x0)
    # print(f"f(x0) = {f_x0}")

verify_candidates = False
if verify_candidates:
    import numpy as np
    from s2mpjlib import *

    def has_fixed_variables(pb_instance):
        """
        VÃ©rifier si un problÃ¨me a des variables fixÃ©es
        
        Returns:
            True si xlower[i] == xupper[i] pour au moins une variable
            False sinon
        """
        bounds_low = np.asarray(pb_instance.xlower, dtype=float).flatten()
        bounds_high = np.asarray(pb_instance.xupper, dtype=float).flatten()
        
        fixed_vars = np.where(bounds_low == bounds_high)[0]
        return len(fixed_vars) > 0


    # Liste des 30 problÃ¨mes Ã  vÃ©rifier
    candidates = [
        'BQPGABIM', 'CHEBYQAD', 'EXPLIN', 'EXPLIN2', 'EXPQUAD',
        'HADAMARD', 'HS44', 'LINVERSE', 'MCCORMCK', 'NCVXBQP1',
        'NCVXBQP2', 'NCVXBQP3', 'NONSCOMP', 'PENALTY2', 'QRTQUAD',
        'SINEALI', 'SVANBERG', 'TORSION1', 'TORSION2', 'BIGGSB1',
        'BQP1VAR', 'CVXQP1', 'NCVXQP1', 'NCVXQP2', 'NCVXQP3',
        'KOEBHELB', 'S368', 'WEEDS', 'YFIT', 'NCVXQP4',    'ARGLINA', 'CHEBYQAD', 'EXPLIN', 'EXPLIN2', 'COSINE', 'HADAMARD',
        'BDQRTIC', 'LINVERSE', 'MCCORMCK', 'NCVXBQP1', 'NCVXBQP2', 'NCVXBQP3',
        'NONSCOMP', 'PENALTY2', 'BROYDN3D', 'SINEALI', 'BRATU1D', 'TORSION1',
        'TORSION2', 'BIGGSB1', 'CATENARY', 'CVXQP1', 'NCVXQP1', 'NCVXQP2',
        'NCVXQP3', 'CYCLOOCT', 'S368', 'NCVXQP4', 'CHANDHEU', 'CBRATU3D',
        'ROSENBR', 'TORSION1', 'POWELLSG', 'EXPQUAD',  'QRTQUAD'
    ]
    candidates = list(set(candidates))  # Supprimer les doublons

    # Tester chaque problÃ¨me
    problems_without_fixed = []
    problems_with_fixed = []

    for name in candidates:
        try:
            # CrÃ©er une instance avec paramÃ¨tre par dÃ©faut (gÃ©nÃ©ralement n=10)
            ProblemClass = globals()[name]
            
            # Essayer avec n=10 ou sans paramÃ¨tre
            try:
                pb = ProblemClass(10)
            except TypeError:
                try:
                    pb = ProblemClass()
                except TypeError:
                    print(f"âš ï¸  {name}: Impossible Ã  instancier")
                    continue
            
            if has_fixed_variables(pb):
                problems_with_fixed.append(name)
                print(f"âŒ {name:15} | HAS fixed variables")
            else:
                problems_without_fixed.append(name)
                print(f"âœ“  {name:15} | ALL variables free")
        
        except Exception as e:
            print(f"âš ï¸  {name:15} | Error: {str(e)[:50]}")

    print(f"\n{'='*60}")
    print(f"âœ“  SANS variables fixÃ©es ({len(problems_without_fixed)}): ")
    print(f"   {problems_without_fixed}")
    print(f"\nâŒ AVEC variables fixÃ©es ({len(problems_with_fixed)}): ")
    print(f"   {problems_with_fixed}")
    print(f"{'='*60}")

    # Liste finale Ã  utiliser
    FINAL_PROBLEMS = problems_without_fixed
    print(f"\nâœ… LISTE FINALE ({len(FINAL_PROBLEMS)} problÃ¨mes):")
    for i, name in enumerate(FINAL_PROBLEMS, 1):
        print(f"   {i:2d}. {name}")