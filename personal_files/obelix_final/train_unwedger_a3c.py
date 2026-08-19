import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import numpy as np
import time

ACTIONS = ["L45","L22","FW","R22","R45"]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(18,128), nn.ReLU())
        self.actor = nn.Linear(128,5)
        self.critic = nn.Linear(128,1)

    def forward(self,x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

def load_env(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UnwedgeEnv

def worker(global_net, opt, env_fn, args):
    local_net = Net()
    local_net.load_state_dict(global_net.state_dict())

    env = env_fn(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        difficulty=args.difficulty,
        max_steps=args.max_steps,
        wall_obstacles=True,
        box_speed=args.box_speed,
        seed=args.seed
        )

    for ep in range(args.episodes):
        s = env.reset()
        done = False

        states, actions, rewards = [],[],[]

        while not done:
            st = torch.tensor(s).float().unsqueeze(0)
            logits,v = local_net(st)

            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample().item()

            s2,r,done = env.step(ACTIONS[a])

            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = s2

            if done or len(states)>20:
                R = 0
                returns = []
                for rwd in reversed(rewards):
                    R = rwd + args.gamma*R
                    returns.insert(0,R)

                returns = torch.tensor(returns)

                logits,values = local_net(torch.tensor(states).float())
                dist = torch.distributions.Categorical(logits=logits)

                logp = dist.log_prob(torch.tensor(actions))
                adv = returns - values.squeeze()

                loss = -(logp*adv.detach()).mean() + adv.pow(2).mean()

                opt.zero_grad()
                loss.backward()

                for gp,lp in zip(global_net.parameters(), local_net.parameters()):
                    gp._grad = lp.grad

                opt.step()
                local_net.load_state_dict(global_net.state_dict())

                states,actions,rewards = [],[],[]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=256)

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.1)
    ap.add_argument("--eps_decay", type=int, default=1000000)

    ap.add_argument("--checkpoint", default="pusher.pth")
    ap.add_argument("--resume_checkpoint", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    UnwedgeEnv = load_env(args.obelix_py)

    global_net = Net()
    global_net.share_memory()

    opt = optim.Adam(global_net.parameters(), lr=args.lr)

    procs = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=worker, args=(global_net,opt,UnwedgeEnv, args))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    torch.save(global_net.state_dict(), args.checkpoint)

if __name__=="__main__":
    main()