"""
Module utilitaire pour charger dynamiquement les classes de problèmes S2MPJ
et reconstruire des instances de problèmes à partir d'un fichier JSON.
"""

import os
import sys
import json
import importlib
import inspect

import numpy as np


def setup_s2mpj_path(s2mpj_root=None):
    """
    Ajoute les dossiers S2MPJ nécessaires à sys.path.

    Parameters
    ----------
    s2mpj_root : str, optional
        Chemin racine contenant le dossier "S2MPJ". Si None, utilise la
        variable d'environnement S2MPJ_ROOT, avec un fallback par défaut.
    """
    if s2mpj_root is None:
        S2MPJ_ROOT = r"C:\Users\adurand\dev\OptimPoblem"

    main_dir = os.path.join(S2MPJ_ROOT, "S2MPJ")
    # main_dir = os.path.join(s2mpj_root, "S2MPJ")
    dir2 = os.path.join(main_dir, "python_problems")

    for path in (main_dir, dir2):
        if path not in sys.path:
            sys.path.insert(0, path)


_CLASS_CACHE = {}


def get_s2mpj_class(name):
    """
    Importe dynamiquement (et met en cache) la classe S2MPJ `name`.

    Parameters
    ----------
    name : str
        Nom du problème S2MPJ (ex: "ARGLINA"). Doit correspondre au nom
        du module .py et de la classe qu'il contient.

    Returns
    -------
    type : la classe du problème, prête à être instanciée.
    """
    if name not in _CLASS_CACHE:
        module = importlib.import_module(name)
        cls = getattr(module, name, None)
        if cls is None or not inspect.isclass(cls):
            raise ImportError(
                f"Classe '{name}' introuvable dans le module '{name}'"
            )
        _CLASS_CACHE[name] = cls
    return _CLASS_CACHE[name]


def load_problems_from_json(filepath, s2mpj_root=None):
    """
    Charge un fichier JSON de configurations de problèmes et reconstruit
    les instances S2MPJ correspondantes.

    Parameters
    ----------
    filepath : str
        Chemin vers le fichier JSON contenant la liste des problèmes.
    s2mpj_root : str, optional
        Racine S2MPJ (voir setup_s2mpj_path). Configuré automatiquement
        au premier appel si non déjà fait.

    Returns
    -------
    list of dict
        Chaque dict contient (au minimum) les clés du JSON d'origine,
        plus 'instance' (objet S2MPJ instancié) et les tableaux numpy
        reconvertis ('initial_doe', 'x0', 'shift', 'bounds_low',
        'bounds_high').
    """
    setup_s2mpj_path(s2mpj_root)

    with open(filepath, "r") as f:
        problems_data = json.load(f)

    problems = []
    for p in problems_data:
        name = p['name']
        param = p['param']

        try:
            pb_class = get_s2mpj_class(name)
        except ImportError as e:
            print(f"⚠️ Classe {name} non trouvée : {e}")
            continue

        try:
            if isinstance(param, list):
                instance = pb_class(*param)
            else:
                instance = pb_class(param)
        except Exception as e:
            print(f"❌ Erreur {name}: {e}")
            continue

        p['instance'] = instance
        p['initial_doe'] = np.array(p['initial_doe'])
        p['x0'] = np.array(p['x0'])
        p['shift'] = np.array(p['shift'])
        p['bounds_low'] = np.array(p['bounds_low'])
        p['bounds_high'] = np.array(p['bounds_high'])

        problems.append(p)

    print(f"✓ {len(problems)} problèmes chargés")
    return problems

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "problems_list.json")

    problems = load_problems_from_json(json_path)
    for p in problems:
        instance = p['instance']
        # Debug : affiche les attributs disponibles
        print(f"Problème: {p['name']}, attributs: {[a for a in dir(instance) if not a.startswith('_')]}")
        break  # juste pour le premier, histoire d'inspecter