import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "pusher.pth") 

    import torch
    import torch.nn as nn

    class PPO(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(18, 128),
                nn.ReLU()
            )
            self.actor = nn.Linear(128, 5)
            self.critic = nn.Linear(128, 1)

        def forward(self, x):
            x = self.shared(x)
            return self.actor(x), self.critic(x)

    model = PPO()

    ckpt = torch.load(wpath, map_location="cpu")
    model.load_state_dict(ckpt)

    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits, _ = _MODEL(x)
        logits = logits.squeeze(0).numpy()

    # deterministic action
    # action = int(np.argmax(logits))

    # stochastic action
    probs = np.exp(logits) / np.sum(np.exp(logits))
    action = rng.choice(len(ACTIONS), p=probs)

    return ACTIONS[action]