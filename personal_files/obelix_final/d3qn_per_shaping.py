import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
from collections import deque
from dataclasses import dataclass
import time

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(4)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.reset_parameters(sigma_init)

    def reset_parameters(self, sigma_init):
        mu_range = 1 / np.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init * mu_range)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init * mu_range)

    def forward(self, x):
        eps_in = torch.randn(self.in_features, device=x.device)
        eps_out = torch.randn(self.out_features, device=x.device)

        f_in = eps_in.sign() * eps_in.abs().sqrt()
        f_out = eps_out.sign() * eps_out.abs().sqrt()

        weight_eps = torch.ger(f_out, f_in)
        bias_eps = f_out

        weight = self.weight_mu + self.weight_sigma * weight_eps
        bias = self.bias_mu + self.bias_sigma * bias_eps

        return torch.nn.functional.linear(x, weight, bias)


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


class PER:
    def __init__(self, cap=100000, alpha=0.6):
        self.cap = cap
        self.alpha = alpha
        self.buf = []
        self.priorities = []
        self.pos = 0

    def add(self, transition, priority=1.0):
        if len(self.buf) < self.cap:
            self.buf.append(transition)
            self.priorities.append(priority)
        else:
            self.buf[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.cap

    def sample(self, batch, beta=0.4):
        probs = np.asarray(self.priorities, dtype=np.float32)
        probs = probs ** self.alpha
        probs /= probs.sum()

        idx = np.random.choice(len(self.buf), batch, p=probs)
        buf = self.buf
        samples = [buf[i] for i in idx]

        weights = (len(self.buf) * probs[idx]) ** (-beta)
        weights /= weights.max()

        return samples, idx, weights

    def update(self, idx, priorities):
        for i, p in zip(idx, priorities):
            self.priorities[i] = p

    def __len__(self):
        return len(self.buf)


class NStep:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buf = deque()

    def add(self, transition):
        self.buf.append(transition)

    def get(self):
        R = 0
        for i, t in enumerate(self.buf):
            R += (self.gamma ** i) * t.r
        return R

    def pop(self):
        return self.buf.popleft()

    def __len__(self):
        return len(self.buf)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


def projection_progress(old_pos, new_pos, target):
    dx = target[0] - old_pos[0]
    dy = target[1] - old_pos[1]
    norm = np.sqrt(dx*dx + dy*dy) + 1e-6

    ux, uy = dx/norm, dy/norm
    mx = new_pos[0] - old_pos[0]
    my = new_pos[1] - old_pos[1]

    return mx * ux + my * uy


def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# Main:
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)


    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=128)

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.1)
    ap.add_argument("--eps_decay", type=int, default=1000000)

    ap.add_argument("--shaping_alpha", type=float, default=1.0)
    ap.add_argument("--per_alpha", type=float, default=0.6)
    ap.add_argument("--nstep", type=int, default=10)

    ap.add_argument("--checkpoint", type=str, default="checkpoint.pth")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--render", action="store_true")

    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    q = DuelingLSTM()
    tgt = DuelingLSTM()
    tgt.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = PER(alpha=args.per_alpha)
    nstep = NStep(args.nstep, args.gamma)

    steps = 0
    best_return = -1e9

    if args.resume:
        try:
            ckpt = torch.load(args.checkpoint)
            q.load_state_dict(ckpt["q"])
            tgt.load_state_dict(ckpt["tgt"])
            opt.load_state_dict(ckpt["opt"])
            steps = ckpt["steps"]
            print("Resumed training from checkpoint")
        except:
            print("No checkpoint found, starting fresh")

    def eps(t):
        return max(args.eps_end,
                   args.eps_start - t/args.eps_decay)

    env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed,
        )

    start_time = time.time()
    episode_times = []

    consecutive_turns = 0

    h = None

    for ep in range(args.episodes):
        ep_start = time.time()
        s = env.reset()

        prev_stuck = 0
        ep_ret = 0

        for _ in range(args.max_steps):

            if random.random() < eps(steps):
                a = random.randint(0, 4)
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(s).float().to(device).unsqueeze(0)
                    qs, h = q(s_t, h)
                a = int(torch.argmax(qs))

            # with torch.no_grad(): 
            #     s_t = torch.from_numpy(s).float().unsqueeze(0) 
            #     qs, h = q(s_t, h) 
            #     a = int(torch.argmax(qs))


            bot_old = (env.bot_center_x, env.bot_center_y)
            box_old = (env.box_center_x, env.box_center_y)

            s2, r_env, done = env.step(ACTIONS[a], render=args.render)

            bot_new = (env.bot_center_x, env.bot_center_y)
            box_new = (env.box_center_x, env.box_center_y)

            r = -1

            if a != 2:
                consecutive_turns += 1
                if consecutive_turns >= 8:
                    r -= 50

            else:
                consecutive_turns = 0

            stuck = s2[17]
            if stuck:
                r -= 100

            if prev_stuck == 1 and stuck == 0:
                r += 100

            prev_stuck = stuck

            if r_env >= 0:
                r += r_env + 1

            if not env.enable_push:
                progress = projection_progress(bot_old, bot_new, box_old)
                r += args.shaping_alpha * progress
                if a == 2 and progress < 0:  # forward
                    r -= 100

            else:
                bx, by = box_old

                boundary = min([
                    (bx, 10),
                    (bx, env.frame_size[0]-10),
                    (10, by),
                    (env.frame_size[1]-10, by)
                ], key=lambda p: (p[0]-bx)**2 + (p[1]-by)**2)

                progress = projection_progress(box_old, box_new, boundary)

                tx, ty = boundary
                ax, ay = bot_new

                v1 = np.array([tx - bx, ty - by])
                v1 = v1 / (np.linalg.norm(v1) + 1e-6)

                v2 = np.array([bx - ax, by - ay])
                v2 = v2 / (np.linalg.norm(v2) + 1e-6)

                alignment = np.dot(v1, v2)

                r += args.shaping_alpha * progress

                r += 0.5 * alignment

                if alignment < 0:
                    r -= 1.0

            ep_ret += r

            nstep.add(Transition(s, a, r, s2, done))

            if len(nstep) >= args.nstep:
                R = nstep.get()
                t0 = nstep.pop()
                replay.add(Transition(t0.s, t0.a, R, s2, done))

            if h is not None:
                h = (h[0].detach(), h[1].detach())

            s = s2
            steps += 1

            state_buf = np.zeros((args.batch, 18), dtype=np.float32)
            next_state_buf = np.zeros((args.batch, 18), dtype=np.float32)
            action_buf = np.zeros(args.batch, dtype=np.int64)
            reward_buf = np.zeros(args.batch, dtype=np.float32)
            done_buf = np.zeros(args.batch, dtype=np.float32)

            if len(replay) > args.batch and steps % 2 == 0:

                samples, idx, weights = replay.sample(args.batch)

                for i, t in enumerate(samples):
                    state_buf[i] = t.s
                    next_state_buf[i] = t.s2
                    action_buf[i] = t.a
                    reward_buf[i] = t.r
                    done_buf[i] = t.done

                sb = torch.as_tensor(state_buf, dtype=torch.float32)
                s2b = torch.as_tensor(next_state_buf, dtype=torch.float32)

                ab = torch.as_tensor(action_buf, dtype=torch.long)
                rb = torch.as_tensor(reward_buf, dtype=torch.float32)
                db = torch.as_tensor(done_buf, dtype=torch.float32)

                w = torch.as_tensor(weights, dtype=torch.float32)

                with torch.no_grad():
                    next_q, _ = q(s2b)
                    next_a = next_q.argmax(1)
                    tgt_q, _ = tgt(s2b)
                    next_val = tgt_q.gather(1, next_a.unsqueeze(1)).squeeze()

                    y = rb + args.gamma * (1-db) * next_val

                pred, _ = q(sb)
                pred = pred.gather(1, ab.unsqueeze(1)).squeeze()

                td = y - pred
                loss = (w * td.pow(2)).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                replay.update(idx, (td.abs().detach().numpy() + 1e-5))

            if steps % 3000 == 0:
                tgt.load_state_dict(q.state_dict())

            # print(
            #     f"Step {steps} | "
            #     f"Action: {ACTIONS[a]} | "
            #     f"Reward: {r:.4f} | "
            #     f"Env: {r_env:.2f} | "
            #     f"Shaping: {args.shaping_alpha * progress:.4f} | "
            # )

            if done:
                break

        ep_time = time.time() - ep_start
        episode_times.append(ep_time)

        avg_time = sum(episode_times[-20:]) / min(len(episode_times), 20)
        remaining_eps = args.episodes - (ep + 1)
        eta = remaining_eps * avg_time

        total_time = time.time() - start_time

        print(
            f"[Ep {ep+1}/{args.episodes}] "
            f"Return: {ep_ret:.2f} | "
            f"Eps: {eps(steps):.3f} | "
            f"Replay: {len(replay)} | "
            f"Time: {ep_time:.2f}s | "
            f"ETA: {eta/60:.1f} min | "
            f"Total: {total_time/60:.1f} min"
        )

        if ep_ret > best_return:
            best_return = ep_ret
            torch.save(q.state_dict(), "best_model_2.0.pth")

        if (ep + 1) % 50 == 0:
            checkpoint_path = f"checkpoint_ep{ep+1}.pth"
            torch.save({
                "q": q.state_dict(),
                "tgt": tgt.state_dict(),
                "opt": opt.state_dict(),
                "steps": steps
            }, checkpoint_path)

            print(f"Checkpoint saved at episode {ep+1}")

    torch.save(q.state_dict(), "final_weights.pth")
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    main()
