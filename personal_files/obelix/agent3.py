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


def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "checkpoint_ep300.pth")

    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, input_dim=18, hidden_dim=64, n_actions=5):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, n_actions)
        
        def forward(self, x, hidden=None):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x, hidden = self.lstm(x, hidden)
            q = self.head(x)
            return q, hidden

    model = Net()
    checkpoint = torch.load(wpath, map_location="cpu")
    if isinstance(checkpoint, dict) and "q_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["q_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    _MODEL = model

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).view(1, 1, -1)

    with torch.no_grad():
        q, _ = _MODEL(x, None)

    qvals = q[0, -1].numpy()

    # if np.random.rand() < 0.5:
    #     action = 2
    # else:
    #     action = int(np.argmax(qvals))
    action = int(np.argmax(qvals))
    return ACTIONS[action]
