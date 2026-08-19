import argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict, Counter
import time

# ACTIONS = ["L45","L22","FW","R22","R45"]
ACTIONS = ["L22", "FW", "R22"]

# Dueling DDQN
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
    
class GroupPERReplay:
    def __init__(self, cap=100000, alpha=0.6):
        self.buf = []
        self.priorities = []
        self.group_keys = []
        self.cap = cap
        self.alpha = alpha
        self.pos = 0

    def add(self, s, a, r, s2, d, priority, group_key):
        item = (s, a, r, s2, d)

        if len(self.buf) < self.cap:
            self.buf.append(item)
            self.priorities.append(float(priority))
            self.group_keys.append(group_key)
        else:
            self.buf[self.pos] = item
            self.priorities[self.pos] = float(priority)
            self.group_keys[self.pos] = group_key
            self.pos += 1
            self.pos %= self.cap

    def sample(self, batch_size, max_per_group=64, beta=0.4):
        groups = defaultdict(list)
        for i, k in enumerate(self.group_keys):
            groups[k].append(i)

        group_keys = list(groups.keys())
        group_scores = np.array([
            max(self.priorities[i] for i in groups[k])
            for k in group_keys
        ], dtype=np.float64)

        group_scores = np.maximum(group_scores, 1e-6)
        group_probs = group_scores ** self.alpha
        group_probs /= group_probs.sum()

        chosen_groups = np.random.choice(
            len(group_keys),
            size=min(len(group_keys), batch_size),
            replace=False,
            p=group_probs
        )

        sampled_indices = []
        per_group_counter = defaultdict(int)

        for gi in chosen_groups:
            k = group_keys[gi]
            idxs = groups[k]
            random.shuffle(idxs)

            for idx in idxs:
                if per_group_counter[k] >= max_per_group:
                    break
                sampled_indices.append(idx)
                per_group_counter[k] += 1
                if len(sampled_indices) >= batch_size:
                    break

            if len(sampled_indices) >= batch_size:
                break

        if len(sampled_indices) < batch_size:
            remaining = [i for i in range(len(self.buf)) if i not in sampled_indices]
            random.shuffle(remaining)
            for idx in remaining:
                k = self.group_keys[idx]
                if per_group_counter[k] >= max_per_group:
                    continue
                sampled_indices.append(idx)
                per_group_counter[k] += 1
                if len(sampled_indices) >= batch_size:
                    break

        batch = [self.buf[i] for i in sampled_indices]
        s, a, r, s2, d = zip(*batch)

        trans_priorities = np.array([self.priorities[i] for i in sampled_indices], dtype=np.float64)
        trans_probs = trans_priorities ** self.alpha
        trans_probs /= trans_probs.sum()

        N = len(self.buf)
        weights = (N * trans_probs) ** (-beta)
        weights /= weights.max() + 1e-8

        return (np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d), sampled_indices, weights)

    def update_priorities(self, idxs, priorities):
        for i, p in zip(idxs, priorities):
            self.priorities[i] = float(max(p, 1e-6))

# taken from the ddqn template code
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# custom function inspired by Weighted Hamming Distance
def phi(obs):
    forward = np.sum(obs[4:12])
    left = np.sum(obs[0:4])
    right = np.sum(obs[12:16])
    ir = obs[16]
    stuck = obs[17]

    score = 0.0
    score += 3.0 * forward
    score -= 1.0 * (left + right)
    score += 5.0 * ir
    score -= 10.0 * stuck

    return score

# Training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", default="finder.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.1)
    ap.add_argument("--eps_decay_steps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_checkpoint", type=str, default=None)
    ap.add_argument("--tau", type=float, default=0.01)
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    q = DuelingDQN()
    tgt = DuelingDQN()
    tgt.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = GroupPERReplay(cap=args.replay)

    gamma = args.gamma
    steps = 0

    state_counts = defaultdict(int)

    def eps(t):
        return max(
            args.eps_end, 
            args.eps_start-(args.eps_start-args.eps_end)*t/(args.eps_decay_steps-1)
            )
    
    # polyak averaging
    def soft_update(tgt, q, tau):
        for tp, qp in zip(tgt.parameters(), q.parameters()):
            tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)

    # compact state representation by grouping directions
    def get_state_key(s):
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()

        forward = int(np.sum(s[4:12]) > 0)
        left = int(np.sum(s[0:4]) > 0)
        right = int(np.sum(s[12:16]) > 0)
        ir = int(s[16])
        stuck = int(s[17])

        return (forward, left, right, ir, stuck)

    # loading from checkpoint if needed
    start_ep = 0
    if args.resume:
        try:
            ckpt = torch.load(args.resume_checkpoint)
            q.load_state_dict(ckpt["q"])
            tgt.load_state_dict(ckpt["tgt"])
            opt.load_state_dict(ckpt["opt"])
            start_ep = ckpt["episode"] + 1
            steps = ckpt["steps"]
            print("Resumed training from checkpoint")
        except:
            print("No checkpoint found, starting fresh")

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        difficulty=args.difficulty,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        box_speed=args.box_speed,
        seed=args.seed
    )

    spin_count = 0
    fw_count = 0

    start_time = time.time()

    for ep in range(start_ep, args.episodes):
        s = env.reset()
        ep_ret = 0

        for _ in range(args.max_steps):

            # decaying eps-greedy
            qs = q(torch.tensor(s).float().unsqueeze(0))
            if random.random() < eps(steps):
                a = random.randint(0, 2)
            else:
                with torch.no_grad():
                    a = qs.argmax().item()
            
            # if steps % 4 == 0:
            #     print(f"Q-values: {np.round(qs.squeeze().detach().numpy(), 2)} | Action: {ACTIONS[a]}")

            s2, r, d = env.step(ACTIONS[a])

            # REWARD SHAPING:

            r += gamma * phi(s2) - phi(s)

            state_change = np.sum(np.abs(s2 - s))
            r += 0.3 * state_change

            # if s2[17] == 1:  # if stuck
            #     r += 150

            # if ACTIONS[a] != "FW":
            #     fw_count = 0
            #     spin_count += 1
            # else:
            #     fw_count += 1
            #     spin_count = 0

            # if spin_count > 7:
            #     r -= 200

            # if fw_count >= 40:
            #     r -= 200

            key = get_state_key(s)
            state_counts[key] += 1
            initial_priority = 1.0

            replay.add(s.copy(), a, r, s2.copy(), d, initial_priority, key)
            s = s2
            ep_ret += r

            if len(replay.buf) > args.batch and steps % 4 == 0:
                # all_keys = [get_state_key(s) for (s, _, _, _, _) in replay.buf]
                # counts = Counter(all_keys)

                # print("REPLAY BUFFER DISTRIBUTION")
                # for k, v in counts.most_common(20):
                #     print(f"{k}: {v}")
                # print("Total:", len(replay.buf))

                sb, ab, rb, s2b, db, idxs, weights = replay.sample(args.batch)

                sb = torch.tensor(sb).float()
                s2b = torch.tensor(s2b).float()
                ab = torch.tensor(ab)
                rb = torch.tensor(rb).float()
                db = torch.tensor(db).float()
                weights = torch.tensor(weights).float()

                # keys = [get_state_key(s) for s in sb]
                # # print(f"State distribution: {Counter(keys)}")
                # counts2 = Counter(keys)
                # for k, v in counts2.most_common(20):
                #     print(f"{k}: {v}")

                with torch.no_grad():
                    next_a = q(s2b).argmax(1)
                    next_q = tgt(s2b).gather(1, next_a.unsqueeze(1)).squeeze()
                    target = rb + gamma * (1 - db) * next_q

                pred = q(sb).gather(1, ab.unsqueeze(1)).squeeze()

                td_error = (target - pred).detach().abs().cpu().numpy()

                new_priorities = []
                for i in range(len(sb)):
                    key = get_state_key(sb[i].cpu().numpy())
                    count = state_counts[key]

                    p = td_error[i] + 1e-5
                    p = min(p, 50)  # clipping

                    new_priorities.append(p)

                replay.update_priorities(idxs, new_priorities)

                loss = (weights * (pred - target) ** 2).mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10)  # gradient clipping
                opt.step()

                soft_update(tgt, q, args.tau)

            steps += 1
            if d: 
                break
        
        # logging and tracking progress
        now = time.time()
        elapsed = now - start_time

        avg_per_ep = elapsed / (ep + 1)
        remaining_eps = args.episodes - ep - 1
        eta = avg_per_ep * remaining_eps

        def format_time(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        print(
            f"EP {ep+1}/{args.episodes} | "
            f"Return: {ep_ret:.1f} | "
            f"Elapsed: {format_time(elapsed)} | "
            f"ETA: {format_time(eta)}"
        )
        zero_state = np.zeros(18, dtype=np.float32)
        zero_tensor = torch.tensor(zero_state).float().unsqueeze(0)

        with torch.no_grad():
            q_zero = q(zero_tensor).squeeze().cpu().numpy()

        print(f"Q(all-zero state): {np.round(q_zero, 3)}")

        stuck_state = np.zeros(18, dtype=np.float32)
        stuck_state[17] = 1

        stuck_tensor = torch.tensor(stuck_state).float().unsqueeze(0)

        with torch.no_grad():
            q_stuck = q(stuck_tensor).squeeze().cpu().numpy()

        print(f"Q(stuck state): {np.round(q_stuck, 3)}")

        # checkpointing every 50 episodes
        if (ep + 1) % 50 == 0:
            checkpoint_path = f"finder_ep{ep+1}.pth"
            torch.save({
                "q": q.state_dict(),
                "tgt": tgt.state_dict(),
                "opt": opt.state_dict(),
                "steps": steps,
                "episode": ep
            }, checkpoint_path)

            print(f"Checkpoint saved at episode {ep+1}")

    # final save after training completion
    torch.save(q.state_dict(), args.out)
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    main()