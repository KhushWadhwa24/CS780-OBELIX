import os
import numpy as np

# ACTIONS = ("L45", "L22", "FW", "R22", "R45")
ACTIONS = ("L22", "FW", "R22")

_MODEL = None


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "finder_ep3950.pth")  # ← change if needed

    import torch
    import torch.nn as nn

    # ---- SAME ARCHITECTURE AS TRAINING ----
    class DuelingDQN(nn.Module):
        def __init__(self, in_dim=18, n_actions=3):
            super().__init__()
            self.base = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())
            self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            self.adv = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))

        def forward(self, x):
            f = self.base(x)
            v = self.value(f)
            a = self.adv(f)
            return v + (a - a.mean(dim=1, keepdim=True))

    model = DuelingDQN()
    model.load_state_dict(torch.load(wpath, map_location="cpu")["q"])
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q_values = _MODEL(x).squeeze(0).numpy()

    # tau = 0.2

    # q_shifted = q_values - np.max(q_values)
    # probs = np.exp(q_shifted / tau)
    # probs /= probs.sum()

    # action = rng.choice(len(q_values), p=probs)

    # action = int(np.argmax(q_values))

    if rng.random() < 0.1:
        action = rng.integers(len(ACTIONS))
    else:
        action = int(np.argmax(q_values))

    return ACTIONS[action]