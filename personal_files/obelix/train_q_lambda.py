import numpy as np
import pickle
import time
from multiprocessing import Pool, cpu_count
from obelix import OBELIX

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

GAMMA          = 0.99
LAMBDA         = 0.3
TOTAL_EPISODES = 500

# ─────────────────────────────────────────────
# Decay Functions
# ─────────────────────────────────────────────
def get_alpha(episode, total_episodes, alpha_start=0.5, alpha_end=0.01):
    rate = (alpha_end / alpha_start) ** (1 / (total_episodes - 1))
    return alpha_start * (rate ** episode)

def get_epsilon(episode, total_episodes, eps_start=0.8, eps_end=0.01):
    rate = (eps_end / eps_start) ** (1 / (total_episodes - 1))
    return eps_start * (rate ** episode)

def obs_to_key(obs):
    return int("".join(str(int(x)) for x in obs), 2)

# ─────────────────────────────────────────────
# Worker Function
# ─────────────────────────────────────────────
def train_worker(args):
    worker_id, ep_start, ep_end, total_episodes, seed = args
    n_eps = ep_end - ep_start

    print(f"[Worker {worker_id}] Starting — episodes {ep_start}→{ep_end} ({n_eps} eps)", flush=True)
    worker_start = time.time()

    rng = np.random.default_rng(seed)

    qA = {}
    qB = {}

    def get_q(table, state):
        if state not in table:
            table[state] = np.zeros(len(ACTIONS))
        return table[state]

    def get_e(table, state):
        if state not in table:
            table[state] = np.zeros(len(ACTIONS))
        return table[state]

    print(f"[Worker {worker_id}] Initialising OBELIX...", flush=True)
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=1000,
        wall_obstacles=True,
        difficulty=0,
        box_speed=2,
        seed=seed,
    )
    print(f"[Worker {worker_id}] Environment ready.", flush=True)

    ep_times = []

    for ep in range(ep_start, ep_end):
        ep_start_time = time.time()

        alpha   = get_alpha(ep, total_episodes)
        epsilon = get_epsilon(ep, total_episodes)

        # clear traces at start of every episode
        eA = {}
        eB = {}

        obs  = env.reset(seed=ep)
        done = False
        total = 0.0
        steps = 0

        while not done:
            state = obs_to_key(obs)

            # epsilon-greedy using averaged Q
            if rng.random() < epsilon:
                a = rng.integers(len(ACTIONS))
            else:
                combined = (get_q(qA, state) + get_q(qB, state)) / 2
                a = int(np.argmax(combined))

            obs2, r, done = env.step(ACTIONS[a], render=False)
            total += r
            steps += 1

            s  = state
            s2 = obs_to_key(obs2)

            if np.random.rand() < 0.5:
                # ── Update A, B evaluates ──────────────
                best_a   = int(np.argmax(get_q(qA, s2)))
                max_q_s2 = 0.0 if done else get_q(qB, s2)[best_a]
                td_error = r + GAMMA * max_q_s2 - get_q(qA, s)[a]

                get_e(eA, s)[a] += 1.0

                for state_k, trace in eA.items():
                    get_q(qA, state_k)[:] += alpha * td_error * trace

                for state_k in list(eA.keys()):
                    eA[state_k] *= GAMMA * LAMBDA
                    if np.max(eA[state_k]) < 1e-6:
                        del eA[state_k]

            else:
                # ── Update B, A evaluates ──────────────
                best_a   = int(np.argmax(get_q(qB, s2)))
                max_q_s2 = 0.0 if done else get_q(qA, s2)[best_a]
                td_error = r + GAMMA * max_q_s2 - get_q(qB, s)[a]

                get_e(eB, s)[a] += 1.0

                for state_k, trace in eB.items():
                    get_q(qB, state_k)[:] += alpha * td_error * trace

                for state_k in list(eB.keys()):
                    eB[state_k] *= GAMMA * LAMBDA
                    if np.max(eB[state_k]) < 1e-6:
                        del eB[state_k]

            obs = obs2

        # ── progress reporting ────────────────
        ep_time = time.time() - ep_start_time
        ep_times.append(ep_time)

        local_ep = ep - ep_start + 1
        if local_ep % 10 == 0:
            avg_ep_time   = sum(ep_times[-10:]) / len(ep_times[-10:])
            eps_remaining = n_eps - local_ep
            eta_str       = time.strftime("%H:%M:%S", time.gmtime(avg_ep_time * eps_remaining))
            elapsed       = time.time() - worker_start
            pct           = 100 * local_ep / n_eps

            print(
                f"[Worker {worker_id}] "
                f"{local_ep:>4}/{n_eps} ({pct:>5.1f}%) | "
                f"reward={total:>8.1f} | "
                f"steps={steps:>5} | "
                f"alpha={alpha:.4f} | "
                f"eps={epsilon:.4f} | "
                f"states={len(qA):>6} | "
                f"traces={len(eA):>5} | "
                f"ep_time={ep_time:.2f}s | "
                f"elapsed={elapsed:.0f}s | "
                f"ETA={eta_str}",
                flush=True
            )

    total_time = time.time() - worker_start
    print(
        f"[Worker {worker_id}] DONE — "
        f"{n_eps} eps in {total_time:.1f}s "
        f"({total_time/n_eps:.2f}s/ep) | "
        f"states={len(qA)}",
        flush=True
    )
    return qA, qB

# ─────────────────────────────────────────────
# Merge Q-tables
# ─────────────────────────────────────────────
def merge_tables(tables):
    print("\nMerging Q-tables from all workers...", flush=True)
    merged_qA = {}
    merged_qB = {}
    counts_A  = {}
    counts_B  = {}

    for i, (qA, qB) in enumerate(tables):
        print(f"  Worker {i} — qA={len(qA)} states, qB={len(qB)} states", flush=True)

        for state, vals in qA.items():
            if state not in merged_qA:
                merged_qA[state] = np.zeros(len(ACTIONS))
                counts_A[state]  = 0
            merged_qA[state] += vals
            counts_A[state]  += 1

        for state, vals in qB.items():
            if state not in merged_qB:
                merged_qB[state] = np.zeros(len(ACTIONS))
                counts_B[state]  = 0
            merged_qB[state] += vals
            counts_B[state]  += 1

    for state in merged_qA:
        merged_qA[state] /= counts_A[state]
    for state in merged_qB:
        merged_qB[state] /= counts_B[state]

    print(f"  Merge complete — unique states: qA={len(merged_qA)}, qB={len(merged_qB)}", flush=True)
    return merged_qA, merged_qB

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":

    n_workers = cpu_count()

    print(f"{'='*60}")
    print(f"Q(λ) Parallel Training")
    print(f"{'='*60}")
    print(f"CPU cores     : {n_workers}")
    print(f"Total episodes: {TOTAL_EPISODES}")
    print(f"Per worker    : {TOTAL_EPISODES // n_workers}")
    print(f"Lambda        : {LAMBDA}")
    print(f"Gamma         : {GAMMA}")
    print(f"{'='*60}\n")

    chunk = TOTAL_EPISODES // n_workers
    worker_args = [
        (i, i * chunk, (i + 1) * chunk, TOTAL_EPISODES, 42 + i)
        for i in range(n_workers)
    ]

    print(f"Spawning {n_workers} workers...\n", flush=True)
    start = time.time()

    with Pool(processes=n_workers) as pool:
        results = pool.map(train_worker, worker_args)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"All workers done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    merged_qA, merged_qB = merge_tables(results)

    print("\nSaving qlambda_table.pkl...", flush=True)
    with open("qlambda_table.pkl", "wb") as f:
        pickle.dump({"qA": merged_qA, "qB": merged_qB}, f)

    print(f"Saved qlambda_table.pkl")
    print(f"Total time: {elapsed/60:.1f} minutes")