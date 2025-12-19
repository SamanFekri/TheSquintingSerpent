# dqn_agent.py
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, in_channels: int, vision_size: int, lidar_dim: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out = 64 * vision_size * vision_size
        # +1 hunger, +2 smell, +lidar_dim
        self.fc = nn.Sequential(
            nn.Linear(conv_out + 3 + lidar_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, grid, hunger, smell, lidar):
        x = self.conv(grid)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, hunger, smell, lidar], dim=1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s_grid, s_hunger, s_smell, s_lidar, a, r, ns_grid, ns_hunger, ns_smell, ns_lidar, done):
        self.buf.append((s_grid, s_hunger, s_smell, s_lidar, a, r, ns_grid, ns_hunger, ns_smell, ns_lidar, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s_grid, s_hunger, s_smell, s_lidar, a, r, ns_grid, ns_hunger, ns_smell, ns_lidar, done = zip(*batch)
        return (
            np.stack(s_grid),
            np.array(s_hunger, dtype=np.float32).reshape(-1, 1),
            np.stack(s_smell).astype(np.float32),     # (B,2)
            np.stack(s_lidar).astype(np.float32),     # (B,lidar_dim)
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ns_grid),
            np.array(ns_hunger, dtype=np.float32).reshape(-1, 1),
            np.stack(ns_smell).astype(np.float32),
            np.stack(ns_lidar).astype(np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(
        self,
        vision_radius: int,
        lidar_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        device: str | None = None,
    ):
        self.vision_size = 2 * vision_radius + 1
        self.lidar_dim = int(lidar_dim)
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q = DQN(3, self.vision_size, lidar_dim=self.lidar_dim).to(self.device)
        self.tgt = DQN(3, self.vision_size, lidar_dim=self.lidar_dim).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.tgt.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(buffer_size)
        self.grad_steps = 0

    def act(self, grid, hunger, smell, lidar, epsilon=0.1, valid_actions=None):
        if random.random() < epsilon:
            return random.choice(valid_actions) if valid_actions else random.randrange(4)

        with torch.no_grad():
            grid_t = torch.from_numpy(grid).unsqueeze(0).to(self.device).float()
            hunger_t = torch.tensor([[float(hunger)]], device=self.device).float()
            smell_t = torch.from_numpy(smell).unsqueeze(0).to(self.device).float()
            lidar_t = torch.from_numpy(lidar).unsqueeze(0).to(self.device).float()
            qvals = self.q(grid_t, hunger_t, smell_t, lidar_t).squeeze(0).cpu().numpy()

        if valid_actions:
            mask = np.full((4,), -1e9, dtype=np.float32)
            mask[valid_actions] = 0.0
            qvals = qvals + mask

        return int(np.argmax(qvals))

    def remember(self, s, a, r, ns, done):
        s_grid, s_hunger, s_smell, s_lidar = s
        ns_grid, ns_hunger, ns_smell, ns_lidar = ns
        self.replay.push(s_grid, s_hunger, s_smell, s_lidar, a, r, ns_grid, ns_hunger, ns_smell, ns_lidar, done)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None

        s_grid, s_hunger, s_smell, s_lidar, a, r, ns_grid, ns_hunger, ns_smell, ns_lidar, done = self.replay.sample(self.batch_size)

        s_grid_t = torch.from_numpy(s_grid).to(self.device).float()
        s_hunger_t = torch.from_numpy(s_hunger).to(self.device).float()
        s_smell_t = torch.from_numpy(s_smell).to(self.device).float()
        s_lidar_t = torch.from_numpy(s_lidar).to(self.device).float()

        a_t = torch.from_numpy(a).to(self.device).long().unsqueeze(1)
        r_t = torch.from_numpy(r).to(self.device).float().unsqueeze(1)

        ns_grid_t = torch.from_numpy(ns_grid).to(self.device).float()
        ns_hunger_t = torch.from_numpy(ns_hunger).to(self.device).float()
        ns_smell_t = torch.from_numpy(ns_smell).to(self.device).float()
        ns_lidar_t = torch.from_numpy(ns_lidar).to(self.device).float()

        done_t = torch.from_numpy(done).to(self.device).float().unsqueeze(1)

        q_sa = self.q(s_grid_t, s_hunger_t, s_smell_t, s_lidar_t).gather(1, a_t)

        with torch.no_grad():
            max_next = self.tgt(ns_grid_t, ns_hunger_t, ns_smell_t, ns_lidar_t).max(dim=1, keepdim=True)[0]
            target = r_t + self.gamma * max_next * (1.0 - done_t)

        loss = self.loss_fn(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.grad_steps += 1
        return float(loss.item())

    def update_target(self):
        self.tgt.load_state_dict(self.q.state_dict())

    # weights only (play)
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {"model": self.q.state_dict(), "vision_size": self.vision_size, "lidar_dim": self.lidar_dim},
            path,
        )

    def load_model(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["model"])
        self.tgt.load_state_dict(ckpt["model"])

    # full checkpoint (resume training)
    def save_checkpoint(self, path: str, episode: int, best_score: int):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "q": self.q.state_dict(),
                "tgt": self.tgt.state_dict(),
                "opt": self.opt.state_dict(),
                "vision_size": self.vision_size,
                "lidar_dim": self.lidar_dim,
                "grad_steps": self.grad_steps,
                "episode": episode,
                "best_score": best_score,
            },
            path,
        )

    def load_checkpoint(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["q"])
        self.tgt.load_state_dict(ckpt["tgt"])
        self.opt.load_state_dict(ckpt["opt"])
        self.grad_steps = int(ckpt.get("grad_steps", 0))
        episode = int(ckpt.get("episode", 0))
        best_score = int(ckpt.get("best_score", -1))
        return episode, best_score
