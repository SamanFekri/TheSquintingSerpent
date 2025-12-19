# Heuristic DQN Snake

Baseline Deep Q-Learning agent that relies on **local vision + food smell** to learn without any built-in path planning.

> 🧭 **Design note**: The heuristic agent sees only a small window around its head, forcing first-person play and preventing overfitting
> to the global map layout.

## Feature summary
- ✅ 3-channel local vision around the head
- ✅ Hunger scalar for reward shaping
- ✅ Food “smell” (relative dx, dy)
- ✅ Compact CNN encoder thanks to the small vision window (fast to train, good performance)
- ❌ Lidar beams
- ❌ BFS-based reward shaping

## Observation space
- 3-channel grid around the head `(2N+1)x(2N+1)` with walls, snake body, and food.
- Hunger scalar in `[0, 1]` derived from steps since the last meal.
- Smell vector `(dx, dy)` pointing from the head to the food using wrap-aware shortest distance.

## Training
Run from inside `heuristic/` so the map path resolves correctly:
```bash
cd heuristic
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
cd heuristic
python play.py \
  --map ../maps/map_10x10.txt \
  --N 2 \
  --wrap \
  --model models/best.pt
```
Add `--human` to control the snake yourself.
