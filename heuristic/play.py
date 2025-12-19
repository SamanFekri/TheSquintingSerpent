# play.py
import argparse
import time

from snake_game import SnakeEnv, SnakeRenderer
from dqn_agent import DQNAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/best.pt")
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--no-wrap", dest="wrap", action="store_false")
    ap.set_defaults(wrap=True)

    ap.add_argument("--map", type=str, required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--human", action="store_true")
    args = ap.parse_args()

    env = SnakeEnv.from_map_file(args.map, vision_radius=args.N, max_hunger=200, wrap=args.wrap)
    renderer = SnakeRenderer(env, cell_size=24, fps=args.fps)

    agent = None
    if not args.human:
        agent = DQNAgent(vision_radius=args.N)
        agent.load(args.model)

    obs = env.reset()
    done = False

    while True:
        if renderer.poll_quit():
            renderer.close()
            return

        if args.human:
            action = renderer.get_human_action(env.direction)
        else:
            grid, hunger, smell = obs
            action = agent.act(grid, hunger, smell, epsilon=0.0)

        obs, reward, done, info = env.step(action)

        renderer.set_status([
            f"Mode: {'HUMAN' if args.human else 'AI'}",
            f"Model: {args.model if not args.human else '-'}",
        ])
        renderer.draw()

        if done:
            time.sleep(0.6)
            obs = env.reset()
            done = False


if __name__ == "__main__":
    main()
