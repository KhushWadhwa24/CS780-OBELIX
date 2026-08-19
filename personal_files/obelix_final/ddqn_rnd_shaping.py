from __future__ import annotations
import argparse, random, time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(8)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class DRQN(nn.Module):
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


class RNDModel(nn.Module):
    def __init__(self, input_dim=18, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ShapedRewardWrapper: 
    def __init__(self, env, alpha: float = 0.5, beta: float = 1.0):
        self.env   = env
        self.alpha = alpha
        self.beta  = beta
 
        self._prev_front_dist:    Optional[float] = None
        self._prev_boundary_dist: Optional[float] = None
 
    def _front_pos(self) -> np.ndarray:
        angle_rad = np.deg2rad(self.env.facing_angle)
        fx = self.env.bot_center_x + self.env.bot_radius * np.cos(angle_rad)
        fy = self.env.bot_center_y + self.env.bot_radius * np.sin(angle_rad)
        return np.array([fx, fy], dtype=np.float32)
 
    def _box_center(self) -> np.ndarray:
        return np.array(
            [self.env.box_center_x, self.env.box_center_y], dtype=np.float32
        )
 
    def _front_to_box_dist(self) -> float:
        return float(np.linalg.norm(self._front_pos() - self._box_center()))
 
    def _boundary_dist(self) -> float:
        bx = self.env.box_center_x
        by = self.env.box_center_y
        W  = self.env.frame_size[1]
        H  = self.env.frame_size[0]
        return float(min(bx - 10, W - 10 - bx, by - 10, H - 10 - by))
 
    def reset(self, seed=None) -> np.ndarray:
        obs = self.env.reset(seed=seed)
        self._prev_front_dist    = self._front_to_box_dist()
        self._prev_boundary_dist = self._boundary_dist()
        return obs
 
    def step(self, action_str: str, render: bool = False):
        obs, env_reward, done = self.env.step(action_str, render=render)

        shaping = 0.0


        # trying some changes:
        if env_reward > 0.0:
            env_reward = env_reward * 10
        if env_reward == -1.0 and action_str == "FW":
            env_reward = 0.0


        in_push = self.env.enable_push
 
        if not in_push:
            curr_front_dist = self._front_to_box_dist()
            if self._prev_front_dist is not None:
                delta    = self._prev_front_dist - curr_front_dist
                if action_str == "FW":
                    shaping += 0.2
                    if delta > 0:
                        shaping += self.alpha * delta * 2
                    else:
                        shaping += 0.0
                else:
                    shaping -= 0.2   # no reward for rotation
                
                
            self._prev_front_dist    = curr_front_dist
            self._prev_boundary_dist = self._boundary_dist()
 
        else:
            bot_pos = np.array(
                [self.env.bot_center_x, self.env.bot_center_y],
                dtype=np.float32
            )
            box_pos = self._box_center()

            vec_to_box = box_pos - bot_pos
            vec_to_box /= np.linalg.norm(vec_to_box)

            angle_rad = np.deg2rad(self.env.facing_angle)
            forward_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])

            alignment = np.dot(forward_vec, vec_to_box)
            curr_boundary_dist = self._boundary_dist()
            if self._prev_boundary_dist is not None:
                delta    = self._prev_boundary_dist - curr_boundary_dist
                if alignment > 0.98:
                    shaping += self.beta * delta
                else:
                    shaping += 0.0
            self._prev_boundary_dist = curr_boundary_dist
            self._prev_front_dist    = None
 
        total_reward = env_reward + shaping
        
 
        return obs, total_reward, done, shaping, env_reward


class Replay:
    def __init__(self, cap=100000):
        # self.buf: Deque[Transition] = deque(maxlen=cap)
        self.buf = []

    def add(self, t: Transition):
        self.buf.append(t)

    def sample_sequence(self, batch_size, seq_len):
        batch = []
        for _ in range(batch_size):
            idx = random.randint(0, len(self.buf) - seq_len - 1)
            # seq = list(self.buf)[idx:idx+seq_len]
            seq = self.buf[idx:idx+seq_len]
            batch.append(seq)
        return batch

    def __len__(self):
        return len(self.buf)


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rnd_lr", type=float, default=1e-3)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=2000)
    ap.add_argument("--eps_start", type=float, default=0.5)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--alpha", type=float, default=0.09)
    ap.add_argument("--beta", type=float, default=0.18)
    ap.add_argument("--eta", type=float, default=50.0)

    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--render", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    raw_env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed,
        )

    env = ShapedRewardWrapper(raw_env, alpha=args.alpha, beta=args.beta)

    q = DRQN()
    tgt = DRQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)

    # RND
    rnd_target = RNDModel()
    rnd_predictor = RNDModel()
    for p in rnd_target.parameters():
        p.requires_grad = False

    rnd_opt = optim.Adam(rnd_predictor.parameters(), lr=args.rnd_lr)

    replay = Replay(cap=args.replay)

    eta = args.eta # scaling factor for intrinsic reward
    steps = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q.to(device)
    tgt.to(device)
    rnd_target.to(device)
    rnd_predictor.to(device)

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        q.load_state_dict(checkpoint['q_state_dict'])
        tgt.load_state_dict(checkpoint['tgt_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'rnd_predictor_state_dict' in checkpoint:
            rnd_predictor.load_state_dict(checkpoint['rnd_predictor_state_dict'])

        if 'rnd_optimizer_state_dict' in checkpoint:
            rnd_opt.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
        start_ep = checkpoint['episode'] + 1
        steps     = checkpoint['steps']
        print(f"Resumed from episode {start_ep}")
    else:
        start_ep = 0

    start_time = time.time()
    last_time = start_time

    # Training loop
    for ep in range(start_ep, args.episodes):

        s = env.reset(seed=args.seed + ep)
        hidden = None
        ep_ret = 0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            with torch.no_grad():
                inp = torch.from_numpy(s.astype(np.float32)).view(1, 1, -1).to(device)
                qvals, hidden = q(inp, hidden)
                
                if np.random.rand() < eps:
                    # if np.random.rand() < 0.5:
                    #     a = 2
                    # else:
                    #     a = np.random.choice([0, 1, 3, 4])
                    # a = np.random.randint(len(ACTIONS))
                    a = 2
                else:
                    a = int(torch.argmax(qvals[0, -1]).item())

            if hidden is not None:
                hidden = (hidden[0].detach(), hidden[1].detach())

            s2, r_env, done, shaping, env_reward_raw = env.step(ACTIONS[a], render=args.render)

            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                target_feat = rnd_target(s_tensor)

            pred_feat = rnd_predictor(s_tensor)
            intrinsic = torch.mean((pred_feat - target_feat)**2).item()
            intrinsic = intrinsic / (1.0 + intrinsic)

            r = r_env + eta * intrinsic
            ep_ret += r

            replay.add(Transition(s, a, r, s2, done))

            s = s2
            steps += 1

            print(
                f"Step {steps} | "
                f"Action: {ACTIONS[a]} | "
                f"Env: {env_reward_raw:.2f} | "
                f"Shape: {shaping:.4f} | "
                f"RND: {intrinsic:.4f} | "
                f"Total: {r:.4f}"
            )

            if steps % 4 == 0 and len(replay) > max(args.warmup, args.batch * args.seq_len):

                sequences = replay.sample_sequence(args.batch, args.seq_len)

                total_loss = 0

                # =======================
                # BATCHED TRAINING (FAST)
                # =======================

                B = len(sequences)
                T = args.seq_len

                # Build batched tensors
                states = torch.from_numpy(
                    np.array([[x.s for x in seq] for seq in sequences], dtype=np.float32)
                ).to(device)

                actions = torch.from_numpy(
                    np.array([[x.a for x in seq] for seq in sequences], dtype=np.int64)
                ).to(device)

                rewards = torch.from_numpy(
                    np.array([[x.r for x in seq] for seq in sequences], dtype=np.float32)
                ).to(device)

                next_states = torch.from_numpy(
                    np.array([[x.s2 for x in seq] for seq in sequences], dtype=np.float32)
                ).to(device)

                dones = torch.from_numpy(
                    np.array([[x.done for x in seq] for seq in sequences], dtype=np.float32)
                ).to(device)

                q_vals, _ = q(states) 
                next_q_online, _ = q(next_states) 
                next_q_target, _ = tgt(next_states)  

                next_actions = torch.argmax(next_q_online, dim=2)  

                next_val = next_q_target.gather(
                    2, next_actions.unsqueeze(-1)
                ).squeeze(-1)

                y = rewards + args.gamma * (1 - dones) * next_val
                pred = q_vals.gather(
                    2, actions.unsqueeze(-1)
                ).squeeze(-1)

                total_loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)

                opt.step()

                all_states = torch.from_numpy(
                    np.array([x.s for seq in sequences for x in seq], dtype=np.float32)
                ).to(device)

                target_feat = rnd_target(all_states)
                pred_feat = rnd_predictor(all_states)

                rnd_loss = torch.mean((pred_feat - target_feat.detach())**2)

                rnd_opt.zero_grad()
                rnd_loss.backward()
                rnd_opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        current_time = time.time()
        episode_time = current_time - last_time
        total_time = current_time - start_time

        progress = (ep + 1) / args.episodes

        if progress > 0:
            ETA = total_time * (1 - progress) / progress
        else:
            ETA = 0

        print(
            f"Ep {ep+1}/{args.episodes} | "
            f"Return: {ep_ret:.2f} | "
            f"Ep Time: {episode_time:.2f}s | "
            f"Total: {total_time/60:.2f}m | "
            f"ETA: {ETA/60:.2f}m | "
            f"Progress: {progress*100:.1f}%"
        )

        if (ep + 1) % 50 == 0:
            checkpoint_path = f"checkpoint_ep{ep+1}.pth"
            torch.save({
                'episode': ep,
                'steps': steps,
                'q_state_dict': q.state_dict(),
                'tgt_state_dict': tgt.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        last_time = current_time

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()