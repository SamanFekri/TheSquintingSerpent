# play.py
import argparse
import csv
import random
import time
from pathlib import Path

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
    ap.add_argument("--model", type=str, default="models/best.pt")
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--hunger_step", type=float, default=1.0, help="Hunger increase per step")
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--no-wrap", dest="wrap", action="store_false")
    ap.set_defaults(wrap=True)

    ap.add_argument("--map", type=str, required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--games", type=int, default=1, help="Number of games to play before exiting")
    ap.add_argument("--log-csv", type=str, help="Path to save per-game results as CSV (score, steps, duration, reason)")
    args = ap.parse_args()

    set_global_seeds(args.seed)

    env = SnakeEnv.from_map_file(
        args.map,
        vision_radius=args.N,
        hunger_step=args.hunger_step,
        wrap=args.wrap,
        seed=args.seed,
    )
    env.use_global_random()
    renderer = SnakeRenderer(env, cell_size=24, fps=args.fps)

    agent = None
    if not args.human:
        agent = DQNAgent(vision_radius=args.N)
        agent.load(args.model)

    obs = env.reset()
    done = False
    game_idx = 1
    steps = 0
    start_time = time.time()
    logs = []

    def record_result(reason: str):
        duration = time.time() - start_time
        logs.append(
            {
                "game": game_idx,
                "score": env.score,
                "steps": steps,
                "duration_sec": round(duration, 3),
                "reason": reason,
            }
        )

    while game_idx <= args.games:
        if renderer.poll_quit():
            renderer.close()
            return

        if args.human:
            action = renderer.get_human_action(env.direction)
        else:
            grid, hunger, smell = obs
            action = agent.act(grid, hunger, smell, epsilon=0.0)

        obs, reward, done, info = env.step(action)
        steps += 1

        renderer.set_status([
            f"Mode: {'HUMAN' if args.human else 'AI'}",
            f"Model: {args.model if not args.human else '-'}",
            f"Game: {game_idx}/{args.games}",
            f"Score: {env.score}",
        ])
        renderer.draw()

        if done:
            record_result(info.get("reason", ""))
            time.sleep(0.6)
            game_idx += 1
            if game_idx > args.games:
                break
            obs = env.reset()
            done = False
            steps = 0
            start_time = time.time()

    renderer.close()

    if args.log_csv and logs:
        csv_path = Path(args.log_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["game", "score", "steps", "duration_sec", "reason"]
            )
            writer.writeheader()
            writer.writerows(logs)

        print(f"Saved game logs to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
