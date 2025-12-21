# play.py
import argparse
import random
import time

import numpy as np
import torch

from snake_game import SnakeEnv, SnakeRenderer
from dqn_agent import DQNAgent


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=str, required=True)
    ap.add_argument("--model", type=str, default="models/best.pt")
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--num_rays", type=int, default=16)
    ap.add_argument("--start_len", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--hunger_step", type=float, default=1.0, help="Hunger increase per step")
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--no-wrap", dest="wrap", action="store_false")
    ap.set_defaults(wrap=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--human", action="store_true")

    # BFS shaping in play mode usually OFF
    ap.add_argument("--bfs", action="store_true")
    ap.add_argument("--no-bfs", dest="bfs", action="store_false")
    ap.set_defaults(bfs=False)

    args = ap.parse_args()

    set_global_seeds(args.seed)

    env = SnakeEnv.from_map_file(
        args.map,
        vision_radius=args.N,
        max_hunger=200,
        hunger_step=args.hunger_step,
        wrap=args.wrap,
        seed=args.seed,
        num_rays=args.num_rays,
        start_length=args.start_len,
        bfs_shaping=args.bfs,
    )
    env.use_global_random()
    renderer = SnakeRenderer(env, cell_size=24, fps=args.fps, padding=90)

    agent = None
    if not args.human:
        lidar_dim = 2 * args.num_rays
        agent = DQNAgent(vision_radius=args.N, lidar_dim=lidar_dim)
        agent.load_model(args.model)

    obs = env.reset()
    done = False

    while True:
        if renderer.poll_quit():
            renderer.close()
            return

        if args.human:
            action = renderer.get_human_action(env.direction)
        else:
            grid, hunger, smell, lidar = obs
            action = agent.act(grid, hunger, smell, lidar, epsilon=0.0)

        obs, reward, done, info = env.step(action)

        renderer.set_status([
            f"Mode: {'HUMAN' if args.human else 'AI'}",
            f"Model: {args.model if not args.human else '-'}",
            f"Rays: {args.num_rays}  Wrap: {args.wrap}",
        ])
        renderer.draw()

        if done:
            time.sleep(0.6)
            obs = env.reset()
            done = False


if __name__ == "__main__":
    main()
