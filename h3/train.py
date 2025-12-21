# train.py
import argparse
import csv
import os
import random

import numpy as np
import torch

from snake_game import SnakeEnv, SnakeRenderer, UP, DOWN, LEFT, RIGHT
from dqn_agent import DQNAgent


def valid_actions_from_dir(direction):
    # prevent reverse (matches env)
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


def log_metrics(csv_path, row, fieldnames):
    if not csv_path:
        return

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--N", type=int, default=1, help="vision radius (1 => 3x3)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--hunger_step", type=float, default=1.0, help="Hunger increase per step")
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--no-wrap", dest="wrap", action="store_false")
    ap.set_defaults(wrap=True)

    ap.add_argument("--map", type=str, required=True, help="path to map txt file (0/1)")
    ap.add_argument("--render", action="store_true", help="show pygame while training")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--save_dir", type=str, default="models")
    ap.add_argument(
        "--log_csv",
        type=str,
        default="",
        help="Path to write per-episode metrics (default: <save_dir>/training_log.csv)",
    )

    args = ap.parse_args()

    set_global_seeds(args.seed)

    env = SnakeEnv.from_map_file(
        args.map,
        vision_radius=args.N,
        max_hunger=200,
        hunger_step=args.hunger_step,
        wrap=args.wrap,
        seed=args.seed,
    )
    env.use_global_random()
    agent = DQNAgent(vision_radius=args.N, lr=1e-3, gamma=0.99, batch_size=64)

    renderer = SnakeRenderer(env, cell_size=24, fps=args.fps) if args.render else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    best_score = -1
    target_update_every = 500
    total_grad_steps = 0

    eps_start, eps_end = 1.0, 0.05
    eps_decay_games = max(1, args.games // 2)

    last_loss = None
    log_csv_path = args.log_csv or os.path.join(args.save_dir, "training_log.csv")
    metric_fields = ["episode", "score", "return", "epsilon", "best_score", "steps", "loss"]

    for ep in range(1, args.games + 1):
        obs = env.reset()
        done = False
        steps = 0
        ep_return = 0.0

        t = min(1.0, ep / eps_decay_games)
        epsilon = eps_start + (eps_end - eps_start) * t

        while not done and steps < args.max_steps:
            if renderer and renderer.poll_quit():
                renderer.close()
                return

            grid, hunger, smell = obs
            valid = valid_actions_from_dir(env.direction)
            action = agent.act(grid, hunger, smell, epsilon=epsilon, valid_actions=valid)

            next_obs, reward, done, info = env.step(action)

            agent.remember(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss
                total_grad_steps += 1
                if total_grad_steps % target_update_every == 0:
                    agent.update_target()

            obs = next_obs
            ep_return += reward
            steps += 1

            last_loss_show = last_loss if last_loss is not None else 0
            if renderer:
                renderer.set_status([
                    f"Episode: {ep}/{args.games}",
                    f"EpisodeScore: {env.score}   BestScore: {best_score}",
                    f"Epsilon: {epsilon:.3f}   Loss: {last_loss_show:.4f}",
                ])
                renderer.draw()

        # save last each episode
        agent.save(last_path)

        # save best by score
        if env.score > best_score:
            best_score = env.score
            agent.save(best_path)

        print(
            f"EP {ep}/{args.games} | score={env.score} | return={ep_return:.2f} "
            f"| eps={epsilon:.3f} | best={best_score}"
        )

        log_metrics(
            log_csv_path,
            {
                "episode": ep,
                "score": env.score,
                "return": ep_return,
                "epsilon": epsilon,
                "best_score": best_score,
                "steps": steps,
                "loss": "" if last_loss is None else float(last_loss),
            },
            metric_fields,
        )

    if renderer:
        renderer.close()

    print(f"\nSaved best model: {best_path}")
    print(f"Saved last model: {last_path}")


if __name__ == "__main__":
    main()
