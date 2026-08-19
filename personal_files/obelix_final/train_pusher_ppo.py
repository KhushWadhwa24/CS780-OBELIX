import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45","L22","FW","R22","R45"]

class PPO(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(18,128), nn.ReLU())
        self.actor = nn.Linear(128,5)
        self.critic = nn.Linear(128,1)

    def forward(self,x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

def load_env(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PusherEnv

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

    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=256)

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.1)
    ap.add_argument("--eps_decay", type=int, default=1000000)

    ap.add_argument("--checkpoint", default="pusher.pth")
    ap.add_argument("--resume_checkpoint", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    PusherEnv = load_env(args.obelix_py)

    agent = PPO()
    opt = optim.Adam(agent.parameters(), lr=args.lr)

    if args.resume:
        try:
            ckpt = torch.load(args.resume_checkpoint)
            agent.load_state_dict(ckpt["agent"])
            opt.load_state_dict(ckpt["opt"])
            steps = ckpt["steps"]
            print("Resumed training from checkpoint")
        except:
            print("No checkpoint found, starting fresh")

    gamma = args.gamma
    steps = 0
    
    def eps(t):
        return max(args.eps_end, args.eps_start - (args.eps_start-args.eps_end) * t / (args.eps_decay-1))

    start_time = time.time()

    env = PusherEnv(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            difficulty=args.difficulty,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            box_speed=args.box_speed,
            seed=args.seed
        )

    for ep in range(args.episodes):
        s = env.reset()

        states, actions, rewards, logps, values = [],[],[],[],[]

        for _ in range(args.max_steps):
            st = torch.tensor(s).float().unsqueeze(0)
            logits,v = agent(st)

            # epsilon = eps(steps)
            dist = torch.distributions.Categorical(logits=logits)

            # if np.random.rand() < epsilon:
            #     a = torch.tensor(np.random.randint(0, 5))
            #     logp = dist.log_prob(a)
            # else:
            #     a = dist.sample()
            #     logp = dist.log_prob(a)

            a = dist.sample()
            logp = dist.log_prob(a)

            s2,r,d = env.step(ACTIONS[a.item()])

            states.append(s)
            actions.append(a)
            rewards.append(r)
            logps.append(logp)
            values.append(v)

            steps += 1

            s = s2
            if d: break

        # returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma*G
            returns.insert(0,G)

        returns = torch.tensor(returns)

        ep_ret = returns[0].item()
        values = torch.stack(values).squeeze()

        adv = returns - values.detach()

        logits,_ = agent(torch.from_numpy(np.array(states)).float())
        dist = torch.distributions.Categorical(logits=logits)

        actions = torch.stack(actions).squeeze()
        new_logps = dist.log_prob(actions)
        ratio = torch.exp(new_logps - torch.stack(logps))

        surr1 = ratio*adv
        surr2 = torch.clamp(ratio,0.8,1.2)*adv

        loss = -torch.min(surr1,surr2).mean() + (returns-values).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

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

        if ep%50==0:
            print("EP",ep,"Return",sum(rewards))

    torch.save(agent.state_dict(), args.checkpoint)

if __name__=="__main__":
    main()