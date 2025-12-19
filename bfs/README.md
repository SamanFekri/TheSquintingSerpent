# BFS-Shaped Lidar DQN Snake

DQN policy that keeps the lidar-enhanced observation space but adds **BFS-based reward shaping** to nudge the agent toward safer routes.

## Observation space
- Local grid `(2N+1)x(2N+1)` with walls, body, food (3 channels).
- Hunger scalar in `[0, 1]`.
- Smell vector `(dx, dy)` toward the food.
- Lidar vector `(2 * num_rays,)` with normalized distances to the body and to walls.

## Reward shaping (environment-side)
- **BFS to food**: when no path exists, apply `--bfs_space_penalty` (default `0.5`).
- **BFS to tail**: when the tail is unreachable, apply `--bfs_tail_penalty` (default `0.5`).
- **Custom death penalty** via `--death_penalty` (default `5.0`).

The BFS results are *not* given to the agent—only the reward signal changes.

## Training
Run from inside `bfs/`:
```bash
cd bfs
python train.py \
  --map ../maps/map_10x10.txt \
  --N 2 \             # vision radius
  --num_rays 16 \      # lidar beams
  --bfs \              # enable BFS shaping (default)
  --bfs_space_penalty 0.5 \    # unreachable-food penalty
  --bfs_tail_penalty 0.5 \     # unreachable-tail penalty
  --death_penalty 5.0 \        # extra loss on death
  --games 2000
```

## Playing
```bash
cd bfs
python play.py \
  --map ../maps/map_10x10.txt \
  --N 2 \
  --num_rays 16 \
  --wrap \
  --model models/best.pt
```
Add `--human` to take control manually.
