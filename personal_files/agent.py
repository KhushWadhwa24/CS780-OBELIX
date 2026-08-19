"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (weights.pth) inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation.

The evaluator will import this file and call `policy(obs, rng)`.
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  # stores the loaded model

def _build_model():
    import torch
    import torch.nn as nn

    class DuelingLSTM(nn.Module):
        def __init__(self, in_dim=18, n_actions=5, hidden=64):
            super().__init__()
            self.fc = nn.Linear(in_dim, hidden)
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

            self.value = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

            self.advantage = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )

        def forward(self, x, h=None):
            x = torch.relu(self.fc(x))
            x, h = self.lstm(x.unsqueeze(1), h)
            x = x.squeeze(1)

            V = self.value(x)
            A = self.advantage(x)
            Q = V + (A - A.mean(dim=1, keepdim=True))
            return Q, h

    return DuelingLSTM()

def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "checkpoint_ep.pth")

    import torch
    import torch.nn as nn

    model = _build_model()
    ckpt = torch.load(wpath, map_location="cpu")

    model.load_state_dict(ckpt["q"])

    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
    _load_once()

    import torch
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q_vals, _ = _MODEL(x)
        q_vals = q_vals.squeeze(0).numpy()

    # if rng.random() < 0.2:  # epsilon-greedy exploration
    #     # choose out of 0, 1, 3, 4 (not forward)
    #     action = rng.choice([0, 1, 3, 4])
    # else:
    action = int(np.argmax(q_vals))

    return ACTIONS[action]
