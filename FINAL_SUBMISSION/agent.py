import os
import numpy as np

ACTIONS = ("L22", "FW", "R22")

_MODEL = None
_ACTION_QUEUE = []


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "checkpoint_ep4000.pth")

    import torch
    import torch.nn as nn

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
    global _ACTION_QUEUE
    _load_once()

    import torch
    if _ACTION_QUEUE:
        action = _ACTION_QUEUE.pop(0)
        return ACTIONS[action]

    stuck = int(obs[17]) 

    if stuck == 1:
        _ACTION_QUEUE = [0, 0, 0, 0, 1, 1, 1, 1]
        action = _ACTION_QUEUE.pop(0)
        return ACTIONS[action]

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q_values = _MODEL(x).squeeze(0).numpy()

    if rng.random() < 0.05:
        action = rng.integers(len(ACTIONS))
    else:
        action = int(np.argmax(q_values))

    return ACTIONS[action]