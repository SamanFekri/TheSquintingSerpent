# train.py
import argparse
import os
import random

import numpy as np
import torch

from snake_game import SnakeEnv, SnakeRenderer, UP, DOWN, LEFT, RIGHT
from dqn_agent import DQNAgent


def valid_actions_from_dir(direction):
    # prevent reverse (matches env behavior)
    if direction == UP:
        return [UP, LEFT, RIGHT]
    if direction == DOWN:
        return [DOWN, LEFT, RIGHT]
    if direction == LEFT:
        return [LEFT, UP, DOWN]
    if direction == RIGHT:
        return [RIGHT, UP, DOWN]
    return [UP, DOWN, LEFT, RIGHT]


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=str, required=True)
    ap.add_argument("--games", type=int, default=2000, help="NEW episodes to run")
    ap.add_argument("--N", type=int, default=2, help="vision radius (2 => 5x5)")
    ap.add_argument("--num_rays", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--hunger_step", type=float, default=0.1, help="Hunger increase per step")
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--no-wrap", dest="wrap", action="store_false")
    ap.set_defaults(wrap=True)

    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--save_dir", type=str, default="models")
    ap.add_argument("--resume", type=str, default="", help="path to checkpoint.pt")
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
    )
    lidar_dim = 2 * args.num_rays  # body + wall dists
    agent = DQNAgent(vision_radius=args.N, lidar_dim=lidar_dim, lr=1e-3, gamma=0.99, batch_size=64)

    renderer = SnakeRenderer(env, cell_size=24, fps=args.fps, padding=90) if args.render else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best.pt")
    last_model_path = os.path.join(args.save_dir, "last.pt")
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pt")

    start_episode = 1
    best_score = -1

    if args.resume:
        loaded_ep, loaded_best = agent.load_checkpoint(args.resume)
        start_episode = loaded_ep + 1
        best_score = loaded_best
        agent.update_target()
        print(f"Resumed from {args.resume}: last_episode={loaded_ep}, best_score={best_score}")

    # epsilon schedule (based on absolute episode count)
    eps_start, eps_end = 1.0, 0.05
    eps_decay_episodes = max(1, (start_episode + args.games) // 2)

    end_episode = start_episode + args.games - 1
    target_update_every = 500
    last_loss = None

    for ep in range(start_episode, end_episode + 1):
        obs = env.reset()
        done = False
        steps = 0
        ep_return = 0.0

        t = min(1.0, ep / eps_decay_episodes)
        epsilon = eps_start + (eps_end - eps_start) * t

        while not done and steps < args.max_steps:
            if renderer and renderer.poll_quit():
                renderer.close()
                return

            grid, hunger, smell, lidar = obs
            valid = valid_actions_from_dir(env.direction)
            action = agent.act(grid, hunger, smell, lidar, epsilon=epsilon, valid_actions=valid)

            next_obs, reward, done, info = env.step(action)

            agent.remember(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss
                if agent.grad_steps % target_update_every == 0:
                    agent.update_target()

            obs = next_obs
            ep_return += reward
            steps += 1

            if renderer:
                loss_str = "..." if last_loss is None else f"{last_loss:.6f}"
                renderer.set_status([
                    f"Episode: {ep}/{end_episode}",
                    f"EpisodeScore: {env.score}   BestScore: {best_score}",
                    f"Epsilon: {epsilon:.3f}   Loss: {loss_str}",
                    f"Rays: {args.num_rays} (lidar_dim={lidar_dim})",
                ])
                renderer.draw()

        # save last model (weights)
        agent.save_model(last_model_path)

        # save best model (by score)
        if env.score > best_score:
            best_score = env.score
            agent.save_model(best_model_path)

        # save checkpoint (resume)
        agent.save_checkpoint(checkpoint_path, episode=ep, best_score=best_score)

        print(f"EP {ep}/{end_episode} | score={env.score} | return={ep_return:.2f} | eps={epsilon:.3f} | best={best_score}")

    if renderer:
        renderer.close()

    print(f"\nSaved best model: {best_model_path}")
    print(f"Saved last model: {last_model_path}")
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
