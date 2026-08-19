import os
import numpy as np
import pickle

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  # stores the loaded model


def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "qlambda_table.pkl")

    with open(wpath, "rb") as f:
            data = pickle.load(f)

    _MODEL = {
                k: data["qA"].get(k, np.zeros(5)) + data["qB"].get(k, np.zeros(5))
                for k in set(data["qA"]) | set(data["qB"])
            }


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
    _load_once()

    key = int("".join(str(int(x)) for x in obs), 2)
    if key not in _MODEL:
        return "FW"

    return ACTIONS[int(np.argmax(_MODEL[key]))]