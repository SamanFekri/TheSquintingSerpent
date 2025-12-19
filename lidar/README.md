# Lidar-Enhanced DQN Snake

Deep Q-Learning agent augmented with **lidar rays** that estimate distances to walls and the snake’s own body, giving the policy longer-range situational awareness than the baseline heuristic agent.

> 🧭 **Design note**: Lidar supplements (but does not replace) the constrained first-person vision window so the policy still generalizes beyond a single map layout while reasoning locally like an in-world snake.

## Feature summary
- ✅ 3-channel local vision
- ✅ Hunger scalar for reward shaping
- ✅ Food “smell” (relative dx, dy)
- ✅ Lidar beams for wall/body distance estimates
- ✅ Compact vision encoder (small window) keeps the core CNN lightweight while lidar adds range
- ❌ BFS-based reward shaping

## Observation space
- Local grid `(2N+1)x(2N+1)` with walls, body, food (3 channels).
- Hunger scalar in `[0, 1]`.
- Smell vector `(dx, dy)` pointing to food (wrap-aware shortest path).
- **Lidar vector** `(2 * num_rays,)` = `[body_distances..., wall_distances...]`, each normalized to `[0, 1]` where `1` means nothing hit within range.

## Training
Run from inside `lidar/`:
```bash
cd lidar
python train.py \
  --map ../maps/map_10x10.txt \
  --N 2 \           # vision radius
  --num_rays 16 \    # lidar beams
  --render \         # optional pygame viewer
  --wrap \           # allow wraparound borders
  --games 2000
```

Highlights:
- DQN accepts the lidar vector alongside vision, hunger, and smell (`dqn_agent.py`).
- Replay buffer + target network updates every 500 gradient steps.
- Epsilon anneals from 1.0 → 0.05 across ~half the episodes.

## Playing
```bash
cd lidar
python play.py \
  --map ../maps/map_10x10.txt \
  --N 2 \
  --num_rays 16 \
  --wrap \
  --model models/best.pt
```
Use `--human` to take control yourself.
