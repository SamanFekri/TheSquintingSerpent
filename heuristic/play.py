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
    folder_name = Path(__file__).resolve().parent.name
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
    env.max_hunger = env.width * env.height
    env.use_global_random()
    renderer_fps = args.fps if args.human else max(args.fps, 120)
    renderer = SnakeRenderer(env, cell_size=24, fps=renderer_fps)

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
    csv_file = None
    csv_writer = None
    csv_needs_header = False

    if args.log_csv:
        csv_path = Path(args.log_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
        csv_file = csv_path.open("a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=["game", "score", "steps", "duration_sec", "reason"]
        )
        if csv_needs_header:
            csv_writer.writeheader()

    def record_result(reason: str):
        duration = time.time() - start_time
        entry = {
            "game": game_idx,
            "score": env.score,
            "steps": steps,
            "duration_sec": round(duration, 3),
            "reason": reason,
        }
        logs.append(entry)
        if csv_writer is not None:
            csv_writer.writerow(entry)
            csv_file.flush()
        print(
            f"Game {game_idx}/{args.games} -> score: {env.score}, steps: {steps}, reason: {reason}"
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
            f"Folder: {folder_name}",
            f"Mode: {'HUMAN' if args.human else 'AI'}",
            f"Model: {args.model if not args.human else '-'}",
            f"Game: {game_idx}/{args.games}",
            f"Score: {env.score}",
        ])
        renderer.draw()

        if done:
            record_result(info.get("reason", ""))
            time.sleep(0.6 if args.human else 0.05)
            game_idx += 1
            if game_idx > args.games:
                break
            obs = env.reset()
            done = False
            steps = 0
            start_time = time.time()

    renderer.close()

    if csv_file is not None:
        csv_file.close()
        print(f"Saved game logs to {Path(args.log_csv).resolve()}")


if __name__ == "__main__":
    main()
