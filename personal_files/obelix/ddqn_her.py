"""Offline trainer: Double DQN + replay buffer (CPU) + HER + reward shaping for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python train_dqn.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0 --wall_obstacles

                        ALGORITHM: DOUBLE DEEP Q-NETWORK (DDQN)


Double DQN is one of the most widely used and reliable improvements over the original Deep Q-Network (DQN).

Main problems it solves:
Vanilla DQN often overestimates true action values.
This happens because the same network is used twice:
   1. to pick the best-looking action in the next state (max)
   2. to evaluate how good that action actually is

When Q-values are noisy (which they almost always are early in training),
this double usage creates optimistic bias → the agent thinks some
actions are much better than they really are → leads to unstable learning.

Double DQN solution:
Split the responsibilities:
• Use the online / main Q-network  to SELECT which action looks best
• Use the target Q-network to EVALUATE (give the actual value)

So instead of:

    target = r + γ × max_a Q_target(s', a)

We do:

    target = r + γ × Q_target( s',   argmax_a Q_online(s', a)   )

This small change dramatically reduces overestimation and makes learning
much more stable — especially in environments with large action spaces
or noisy rewards.

For More Details please refer to https://arxiv.org/pdf/1509.06461 .


"""

from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
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
    box_pos: np.ndarray


# Episode Buffer
@dataclass
class EpisodeBuffer:
    transitions: List[Transition] = field(default_factory=list)

    def add(self, t: Transition):
        self.transitions.append(t)

    def clear(self):
        self.transitions.clear()

    def __len__(self): 
        return len(self.transitions)
    

# HER Replay Buffer
class HERReplayBuffer: 
    def __init__(
        self,
        cap: int = 200_000,
        n_her_goals: int = 4,
        goal_threshold: float = 15.0,   
        arena_size: int = 500,
    ):
        self.buf: Deque[Transition] = deque(maxlen=cap)
        self.n_her_goals = n_her_goals
        self.goal_threshold = goal_threshold
        self.arena_size = arena_size

    def _relabeled_reward(
        self, box_pos: np.ndarray, goal: np.ndarray
    ) -> float:
        dist = float(np.linalg.norm(box_pos - goal))
        return 2000.0 if dist < self.goal_threshold else -1.0
 
    def add_episode(self, episode: EpisodeBuffer):
        T = len(episode)
        if T == 0:
            return
 
        transitions = episode.transitions
        box_positions = np.array([t.box_pos for t in transitions])
 
        for t_idx, trans in enumerate(transitions):
            self.buf.append(trans)

            future_pool = list(range(t_idx + 1, T))
            if not future_pool:
                continue
 
            k_indices = random.choices(
                future_pool, k=min(self.n_her_goals, len(future_pool))
            )
 
            for k in k_indices:
                relabeled_goal = box_positions[k]
         

                new_r = self._relabeled_reward(trans.box_pos, relabeled_goal)
                new_done = new_r >= 2000.0
 
                relabeled = Transition(
                    s=trans.s,
                    a=trans.a,
                    r=new_r,
                    s2=trans.s2,
                    done=new_done,
                    box_pos=trans.box_pos,
                )
                self.buf.append(relabeled)

    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
 
        s    = np.stack([it.s    for it in items]).astype(np.float32)
        a    = np.array([it.a    for it in items], dtype=np.int64)
        r    = np.array([it.r    for it in items], dtype=np.float32)
        s2   = np.stack([it.s2   for it in items]).astype(np.float32)
        d    = np.array([it.done for it in items], dtype=np.float32)
 
        return s, a, r, s2, d
 
    def __len__(self):
        return len(self.buf)
    

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
        in_push = self.env.enable_push
 
        if not in_push:
            curr_front_dist = self._front_to_box_dist()
            if self._prev_front_dist is not None:
                delta    = self._prev_front_dist - curr_front_dist
                shaping += self.alpha * delta
            self._prev_front_dist    = curr_front_dist
            self._prev_boundary_dist = self._boundary_dist()
 
        else:
            curr_boundary_dist = self._boundary_dist()
            if self._prev_boundary_dist is not None:
                delta    = self._prev_boundary_dist - curr_boundary_dist
                shaping += self.beta * delta
            self._prev_boundary_dist = curr_boundary_dist
            self._prev_front_dist    = None
 
        total_reward = env_reward + shaping
        box_pos      = self._box_center()
 
        return obs, total_reward, done, box_pos


class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)
    def add(self, t: Transition):
        self.buf.append(t)
    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d
    def __len__(self): return len(self.buf)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=2000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--n_her_goals", type=int, default=4)
    ap.add_argument("--goal_threshold", type=float, default=15.0)

    ap.add_argument("--resume", type=str, default=None)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q = DQN()
    tgt = DQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = HERReplayBuffer(
        cap=args.replay,
        n_her_goals=args.n_her_goals,
        goal_threshold=args.goal_threshold,
        arena_size=args.arena_size,
    )

    steps = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)
    
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

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        q.state_dict(checkpoint['q_state_dict'])
        tgt.load_state_dict(checkpoint['tgt_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_ep = checkpoint['episode'] + 1
        steps     = checkpoint['steps']
        print(f"Resumed from episode {start_ep}")
    else:
        start_ep = 0

    for ep in range(start_ep, args.episodes):
        ep_start = time.time()
        
        s = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        episode = EpisodeBuffer()

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            s2, r, done, box_pos = env.step(ACTIONS[a], render=True)
            ep_ret += float(r)
            episode.add(Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done), box_pos=box_pos))
            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                sb_t = torch.tensor(sb)
                ab_t = torch.tensor(ab)
                rb_t = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t = torch.tensor(db)

                with torch.no_grad():
                    next_q = q(s2b_t)
                    next_a = torch.argmax(next_q, dim=1)
                    next_q_tgt = tgt(s2b_t)
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        replay.add_episode(episode)

        print(f"Episode {ep+1}/{args.episodes}: time={(time.time() - ep_start):.2f}s")

        if ep == 0 or (ep + 1) % 50 == 0: 
            print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)}")

        # Save checkpoint every 50 episodes
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

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
