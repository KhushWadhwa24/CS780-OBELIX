import numpy as np
import pickle
import os
from obelix import OBELIX

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

GAMMA = 0.99
TOTAL_EPISODES = 1000

def get_alpha(episode, total_episodes, alpha_start=0.5, alpha_end=0.01):
    rate = (alpha_end / alpha_start) ** (1 / (total_episodes - 1))
    return alpha_start * (rate ** episode)

def get_epsilon(episode, total_episodes, eps_start=0.8, eps_end=0.01):
    rate = (eps_end / eps_start) ** (1 / (total_episodes - 1))
    return eps_start * (rate ** episode)

def obs_to_key(obs):
    return int("".join(str(int(x)) for x in obs), 2)

class DoubleQLearningAgent:
    def __init__(self):
        self.qA = {}
        self.qB = {}

    def _get_q(self, table, state):
        if state not in table:
            table[state] = np.zeros(len(ACTIONS))
        return table[state]
    
    def _combined_q(self, state):
        return self._get_q(self.qA, state) + self._get_q(self.qB, state) / 2

    def act(self, obs, rng, epsilon):
        state = obs_to_key(obs)
        if rng.random() < epsilon:
            return rng.integers(len(ACTIONS))
        return int(np.argmax(self._combined_q(state)))

    def update(self, obs, a, r, obs2, done, alpha):
        s  = obs_to_key(obs)
        s2 = obs_to_key(obs2)

        if np.random.rand() < 0.5:
            best_a = int(np.argmax(self._get_q(self.qA, s2)))
            max_q_s2 = 0.0 if done else self._get_q(self.qB, s2)[best_a]

            td_target = r + GAMMA * max_q_s2
            td_error  = td_target - self._get_q(self.qA, s)[a]
            self._get_q(self.qA, s)[a] += alpha * td_error

        else:
            best_a = int(np.argmax(self._get_q(self.qB, s2)))
            max_q_s2 = 0.0 if done else self._get_q(self.qA, s2)[best_a]

            td_target = r + GAMMA * max_q_s2
            td_error  = td_target - self._get_q(self.qB, s)[a]
            self._get_q(self.qB, s)[a] += alpha * td_error

    def save(self, path="qtable_double.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"qA": self.qA, "qB": self.qB}, f)

    def load(self, path="qtable_double.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.qA = data["qA"]
            self.qB = data["qB"]


agent = DoubleQLearningAgent()
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

for ep in range(TOTAL_EPISODES):

    alpha   = get_alpha(ep, TOTAL_EPISODES)
    epsilon = get_epsilon(ep, TOTAL_EPISODES)

    print(f"Episode {ep+1:>4}/{TOTAL_EPISODES}")
    obs  = env.reset(seed=ep)
    done = False
    total = 0.0

    while not done:
        a             = agent.act(obs, rng, epsilon)
        obs2, r, done = env.step(ACTIONS[a], render=False)
        agent.update(obs, a, r, obs2, done, alpha)
        obs = obs2
        total += r
    
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1:>4}/{TOTAL_EPISODES} | "
              f"reward={total:>8.1f} | "
              f"alpha={alpha:.4f} | "
              f"eps={epsilon:.4f} | "
              f"states visited={len(agent.qA)}")

agent.save("double_q_table.pkl")
print("Saved double_q_table.pkl")