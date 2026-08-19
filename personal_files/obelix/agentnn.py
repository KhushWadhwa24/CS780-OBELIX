"""
Submission template (NoisyNet version).

Loads trained NoisyNet + LSTM model and runs deterministic inference.
"""

import os
import numpy as np

from d3qn_per_shaping import DuelingLSTM

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_H = None  # LSTM hidden state

def _build_model():
    import torch
    import torch.nn as nn
    import numpy as np
    # ============================
    # NOISY LINEAR
    # ============================
    class NoisyLinear(nn.Module):
        def __init__(self, in_features, out_features, sigma_init=0.5):
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features

            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))

            self.noise = True
            self.reset_parameters(sigma_init)

        def reset_parameters(self, sigma_init):
            mu_range = 1 / np.sqrt(self.in_features)

            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(sigma_init * mu_range)

            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(sigma_init * mu_range)

        def forward(self, x):
            if self.noise:
                eps_in = torch.randn(self.in_features, device=x.device)
                eps_out = torch.randn(self.out_features, device=x.device)

                f_in = eps_in.sign() * eps_in.abs().sqrt()
                f_out = eps_out.sign() * eps_out.abs().sqrt()

                weight_eps = torch.ger(f_out, f_in)
                bias_eps = f_out

                weight = self.weight_mu + self.weight_sigma * weight_eps
                bias = self.bias_mu + self.bias_sigma * bias_eps
            else:
                weight = self.weight_mu
                bias = self.bias_mu

            return torch.nn.functional.linear(x, weight, bias)

    # ============================
    # DUELING LSTM (NOISY)
    # ============================
    class DuelingLSTM(nn.Module):
        def __init__(self, in_dim=18, n_actions=5, hidden=64):
            super().__init__()

            self.fc = NoisyLinear(in_dim, hidden)
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

            self.value = nn.Sequential(
                NoisyLinear(hidden, 64),
                nn.ReLU(),
                NoisyLinear(64, 1)
            )

            self.advantage = nn.Sequential(
                NoisyLinear(hidden, 64),
                nn.ReLU(),
                NoisyLinear(64, n_actions)
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


def _disable_noise(model):
    """Turn off noise for deterministic evaluation."""
    for m in model.modules():
        if hasattr(m, "noise"):
            m.noise = False

def _load_once():
    global _MODEL, _H
    if _MODEL is not None:
        return

    import torch

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "ckpt_ep750.pth")

    model = _build_model()
    ckpt = torch.load(wpath, map_location="cpu")

    # 🔥 KEY FIX: handle both formats safely
    state_dict = ckpt["q"] if "q" in ckpt else ckpt

    model.load_state_dict(state_dict)
    model.eval()

    _disable_noise(model)

    _MODEL = model
    _H = None

import torch

submission_dir = os.path.dirname(__file__)
wpath = os.path.join(submission_dir, "checkpoint_ep1250.pth")

model = _build_model()
ckpt = torch.load(wpath, map_location="cpu")

model.load_state_dict(ckpt["q"])
model.eval()

_disable_noise(model)

_MODEL = model
_H = None


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _H
    _load_once()


    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q_vals, _H = _MODEL(x, _H)
        q_vals = q_vals.squeeze(0).numpy()

    action = int(np.argmax(q_vals))
    return ACTIONS[action]

