# H2 DQN Snake

Variant of the baseline heuristic agent where the **N×N local grid encodes walls, the snake body, and food in a single channel**. Values are chosen to emphasize obstacles vs. rewards:
- `-1.0` → walls (including borders when no wrap)
- `-0.5` → snake body (including the head)
- `1.0` → food
- `0.0` → empty space

The agent still receives the hunger scalar and food “smell” vector.

> 🧭 **Design note**: Like the baseline, H2 intentionally limits its vision to a local window so it learns first-person navigation instead
> of overfitting to full-map patterns.

## Feature summary
- ✅ Single-channel encoded local vision
- ✅ Hunger scalar for reward shaping
- ✅ Food “smell” (relative dx, dy)
- ✅ Compact CNN encoder (single channel + small window) for faster training with solid performance
- ❌ Lidar beams
- ❌ BFS-based reward shaping

## Observation space
- 1-channel grid around the head `(2N+1) x (2N+1)` with the numeric encoding above.
- Hunger scalar in `[0, 1]` derived from steps since the last meal.
- Smell vector `(dx, dy)` pointing from the head to the food using wrap-aware shortest distance.

## Training
Run from inside `h2/` so the map path resolves correctly:
```bash
cd h2
python train.py \
  --map ../maps/map_10x10.txt \
  --N 2 \          # vision radius (2 -> 5x5 window)
  --render \        # optional pygame viewer
  --wrap \          # allow wraparound borders
  --games 2000
```

Key details:
- DQN with convolutional encoder + fully connected head (`dqn_agent.py`).
- Experience replay and target network updates every 500 gradient steps.
- Epsilon decays from 1.0 → 0.05 across ~50% of episodes.

## Playing
Watch the trained agent or play manually:
```bash
cd h2
python play.py \
  --map ../maps/map_10x10.txt \
  --N 2 \
  --wrap \
  --model models/best.pt
```
Add `--human` to control the snake yourself.
