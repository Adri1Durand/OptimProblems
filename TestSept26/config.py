"""
Configuration globale de la campagne de benchmark GCO(MADS) vs MADS seul.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
PROBLEM_SEED = 42          # seed dédiée, UNIQUEMENT pour figer les dimensions n des problèmes scalables
DOE_SEED = 0                # seed du DOE initial (LHS), commun à toutes les méthodes/problèmes
RUN_SEEDS = [0]              # seed(s) pour la répétition des optimisations (1 seed ici, extensible)

# ---------------------------------------------------------------------------
# Budget et DOE
# ---------------------------------------------------------------------------
BUDGET_MAX_SIMPLEX_EVAL = 500       # nombre max de (n+1) évaluations de la fonction objectif
DOE_SIZE = 10                # taille du DOE initial (LHS)

# ---------------------------------------------------------------------------
# Bornes de dimension pour les problèmes scalables
# ---------------------------------------------------------------------------
N_MIN = 10
N_MAX = 49

# ---------------------------------------------------------------------------
# Méthodes testées
# ---------------------------------------------------------------------------
SUBSPACE_DIMENSIONS = [4, 6]   # dimensions de sous-espace pour GCO(MADS, PLS)
SUBSPACE_SELECTION = "PLS"

METHOD_NAMES = [
    "MADS",                         # MADS seul (référence)
    "GCO(MADS,PLS,p4)",             # GCO avec PLS, sous-dimension 4
    "GCO(MADS,PLS,p6)",             # GCO avec PLS, sous-dimension 6
]

# ---------------------------------------------------------------------------
# Chemins de sortie
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
RESULTS_DIR2 = "results_mid"
PROFILES_DIR = "profiles_output"