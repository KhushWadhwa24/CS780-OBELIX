import numpy as np
import pickle
import time
from obelix import OBELIX

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

GAMMA          = 0.99
LAMBDA         = 0.3   # trace decay — how far back to propagate reward
TOTAL_EPISODES = 1000

# ─────────────────────────────────────────────
# Decay Functions
# ─────────────────────────────────────────────
def get_alpha(episode, total_episodes, alpha_start=0.5, alpha_end=0.01):
    rate = (alpha_end / alpha_start) ** (1 / (total_episodes - 1))
    return alpha_start * (rate ** episode)

def get_epsilon(episode, total_episodes, eps_start=0.8, eps_end=0.01):
    rate = (eps_end / eps_start) ** (1 / (total_episodes - 1))
    return eps_start * (rate ** episode)

# ─────────────────────────────────────────────
# State Encoding
# ─────────────────────────────────────────────
def obs_to_key(obs):
    return int("".join(str(int(x)) for x in obs), 2)

# ─────────────────────────────────────────────
# Q(λ) Agent
# ─────────────────────────────────────────────
class QLambdaAgent:
    def __init__(self, lam=LAMBDA):
        self.lam = lam        # λ — trace decay parameter
        self.qA  = {}         # Q-table A
        self.qB  = {}         # Q-table B
        self.eA  = {}         # eligibility traces for A
        self.eB  = {}         # eligibility traces for B

    # ── helpers ──────────────────────────────
    def _get_q(self, table, state):
        """Get Q-values, initialise to 0 if state unseen."""
        if state not in table:
            table[state] = np.zeros(len(ACTIONS))
        return table[state]

    def _get_e(self, table, state):
        """Get eligibility trace, initialise to 0 if state unseen."""
        if state not in table:
            table[state] = np.zeros(len(ACTIONS))
        return table[state]

    def _combined_q(self, state):
        """Average both tables for action selection."""
        return (self._get_q(self.qA, state) + self._get_q(self.qB, state)) / 2

    # ── reset traces at episode start ────────
    def reset_traces(self):
        """
        Clear eligibility traces at the START of every episode.
        Traces only live within one episode — they must not
        carry over between episodes.
        """
        self.eA = {}
        self.eB = {}

    # ── action selection ─────────────────────
    def act(self, obs, rng, epsilon):
        """Epsilon-greedy action selection using averaged Q-tables."""
        state = obs_to_key(obs)
        if rng.random() < epsilon:
            return rng.integers(len(ACTIONS))
        return int(np.argmax(self._combined_q(state)))

    # ── Q(λ) update ──────────────────────────
    def update(self, obs, a, r, obs2, done, alpha):
        """
        Q(λ) Double Q update:

        1. Compute TD error for current transition
        2. Increment trace for visited (state, action)
        3. Update ALL states in trace table, weighted by their trace value
        4. Decay ALL traces by γλ

        This means a reward propagates backwards through the
        entire recent history of visited states, not just one step.
        """
        s  = obs_to_key(obs)
        s2 = obs_to_key(obs2)

        if np.random.rand() < 0.5:
            # ── Update A using B as evaluator ─────────
            best_a   = int(np.argmax(self._get_q(self.qA, s2)))
            max_q_s2 = 0.0 if done else self._get_q(self.qB, s2)[best_a]

            td_error = r + GAMMA * max_q_s2 - self._get_q(self.qA, s)[a]

            # increment trace for this (state, action)
            self._get_e(self.eA, s)[a] += 1.0

            # update ALL states in trace — weighted by their eligibility
            for state, trace in self.eA.items():
                self._get_q(self.qA, state)[:] += alpha * td_error * trace

            # decay ALL traces by γλ
            for state in list(self.eA.keys()):
                self.eA[state] *= GAMMA * self.lam
                # prune near-zero traces to save memory
                if np.max(self.eA[state]) < 1e-6:
                    del self.eA[state]

        else:
            # ── Update B using A as evaluator ─────────
            best_a   = int(np.argmax(self._get_q(self.qB, s2)))
            max_q_s2 = 0.0 if done else self._get_q(self.qA, s2)[best_a]

            td_error = r + GAMMA * max_q_s2 - self._get_q(self.qB, s)[a]

            # increment trace for this (state, action)
            self._get_e(self.eB, s)[a] += 1.0

            # update ALL states in trace — weighted by their eligibility
            for state, trace in self.eB.items():
                self._get_q(self.qB, state)[:] += alpha * td_error * trace

            # decay ALL traces by γλ
            for state in list(self.eB.keys()):
                self.eB[state] *= GAMMA * self.lam
                if np.max(self.eB[state]) < 1e-6:
                    del self.eB[state]

    # ── save / load ──────────────────────────
    def save(self, path="qlambda_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"qA": self.qA, "qB": self.qB}, f)

    def load(self, path="qlambda_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.qA = data["qA"]
            self.qB = data["qB"]

# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
agent = QLambdaAgent(lam=LAMBDA)
rng   = np.random.default_rng(11)

env = OBELIX(
    scaling_factor=5,
    arena_size=500,
    max_steps=1000,
    wall_obstacles=True,
    difficulty=0,
    box_speed=2,
    seed=0,
)

print("="*60)
print(f"Q(λ) Training   λ={LAMBDA}  γ={GAMMA}  episodes={TOTAL_EPISODES}")
print("="*60)

start = time.time()
ep_times = []

for ep in range(TOTAL_EPISODES):

    alpha   = get_alpha(ep, TOTAL_EPISODES)
    epsilon = get_epsilon(ep, TOTAL_EPISODES)

    # IMPORTANT — clear traces at start of every episode
    agent.reset_traces()

    obs  = env.reset(seed=ep)
    done = False
    total = 0.0
    steps = 0
    ep_start = time.time()

    while not done:
        a             = agent.act(obs, rng, epsilon)
        obs2, r, done = env.step(ACTIONS[a], render=False)
        agent.update(obs, a, r, obs2, done, alpha)
        obs    = obs2
        total += r
        steps += 1

    ep_time = time.time() - ep_start
    ep_times.append(ep_time)

    if (ep + 1) % 100 == 0:
        avg_time      = sum(ep_times[-100:]) / len(ep_times[-100:])
        eps_remaining = TOTAL_EPISODES - (ep + 1)
        eta           = avg_time * eps_remaining
        eta_str       = time.strftime("%H:%M:%S", time.gmtime(eta))
        elapsed       = time.time() - start

        print(
            f"Episode {ep+1:>4}/{TOTAL_EPISODES} | "
            f"reward={total:>8.1f} | "
            f"steps={steps:>5} | "
            f"alpha={alpha:.4f} | "
            f"eps={epsilon:.4f} | "
            f"states={len(agent.qA):>6} | "
            f"traces={len(agent.eA):>5} | "
            f"elapsed={elapsed:.0f}s | "
            f"ETA={eta_str}"
        )

total_time = time.time() - start
print(f"\nTraining complete in {total_time/60:.1f} minutes")

agent.save("qlambda_table.pkl")
print("Saved qlambda_table.pkl")