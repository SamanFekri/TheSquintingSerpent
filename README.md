# 🐍 TheSquintingSerpent (Deep Reinforcement Learning)

<p align="center">
  <img src="media/logo.png" alt="Logo" width="256" height="256">
</p>

A fully featured **Snake game with multiple Deep Q-Learning (DQN) variants**, supporting:

- 🧠 Reinforcement Learning (PyTorch)
- 👃 Food “smell” (relative food position)
- 🍴 Hunger mechanism (reward shaping)
- 🧱 Custom maps (from `.txt` files)
- 🪟 Wrap / no-wrap borders
- 🎮 Human play
- 📺 Pygame GUI (centered grid, large window)
- 💾 Save / load / resume training
- 🏆 Best model & last model saving

### 📂 Agent variants in this repo
- **`heuristic/`** – baseline DQN with **3-channel local vision**, **hunger scalar**, and **food “smell” vector**.
- **`h2/`** – same sensors as the baseline but **compresses vision to a single encoded grid channel**.
- **`h3/`** – H2-style sensors with the snake body **linearly encoded from tail (-0.01) to head (-0.99)** for clearer ordering.
- **`lidar/`** – adds **lidar rays** for wall/body distances on top of the baseline observation space.
- **`bfs/`** – keeps the lidar observation space and layers in **BFS-based reward shaping** for safety.

Each folder ships its own README with usage and a quick rundown of the features that specific model uses.

> 🧭 **Design note**: All agents learn from a **limited, first-person vision window** around the snake’s head. Constraining visibility keeps
> policies from overfitting to a single map layout and makes the agent play like a snake would—navigating locally—rather than from the
> third-person, full-map view humans get in classic Snake.
>
> This small observation window also keeps the convolutional encoder **compact (fewer parameters)**, so models train faster while maintaining
> strong, map-agnostic performance.

---

## 📁 Project Structure
```
.
├── heuristic/   # Baseline DQN agent with 3-channel vision, hunger, smell
│   ├── train.py
│   ├── play.py
│   ├── dqn_agent.py
│   └── snake_game.py
├── h2/          # Vision-compressed variant sharing the same sensors
│   ├── train.py
│   ├── play.py
│   ├── dqn_agent.py
│   └── snake_game.py
├── h3/          # Gradient-encoded body values (tail -0.01 → head -0.99)
│   ├── train.py
│   ├── play.py
│   ├── dqn_agent.py
│   └── snake_game.py
├── lidar/       # Adds lidar distance rays to the observation
│   ├── train.py
│   ├── play.py
│   ├── dqn_agent.py
│   └── snake_game.py
├── bfs/         # Lidar inputs plus BFS-based reward shaping
│   ├── train.py
│   ├── play.py
│   ├── dqn_agent.py
│   └── snake_game.py
├── maps/        # Text map files used by all agents
├── media/       # Assets (logo)
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install pygame torch numpy
```

Python 3.9+ recommended.

---

## 🧱 Maps

Maps are simple text files:
- 0 → free cell
- 1 → wall

Rectangular grid

Size of the map defines the game size

--- 

## 🧠 AI Observation Space

Each step the AI receives:

1. Local vision grid (3, 2N+1, 2N+1)
- Channel 0: walls
- Channel 1: snake body
- Channel 2: food
2. Hunger ∈ [0, 1]
3. Smell vector (dx, dy)
- Relative distance from head → food
- Normalized
- Wrap-aware (shortest distance)
```python
Observation = (grid, hunger, smell)
```

---

## 🎯 Actions
```ini
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

⚠️ Reverse direction is blocked (classic Snake behavior).
The snake can never move into its neck, even during training.

---

## ☠️ Game Over Rules

The game ends if the snake:
- ❌ Hits a wall
- ❌ Hits its own body
Hunger does not kill directly — it only adds increasing negative reward.

---

## 🏅 Rewards

- +1.0 → eat food
- −1.0 → die (wall or self)
- −0.01 → per step
- extra penalty proportional to hunger

This encourages:
- faster food seeking
- less wandering
- stable learning

---

## 🧪 Training the AI

Basic training (with GUI)
```bash
python train.py \
  --map maps/map_10x10.txt \
  --N 2 \
  --render \
  --wrap \
  --games 5000
```

No wrap (classic borders)
```bash
python train.py \
  --map maps/map_10x10.txt \
  --N 2 \
  --render \
  --no-wrap \
  --games 5000
```

---

## ▶️ Resume Training (IMPORTANT)

Training automatically saves a full checkpoint.
To continue training from where you stopped:

```bash
python train.py \
  --map maps/map_10x10.txt \
  --N 2 \
  --render \
  --wrap \
  --games 2000 \
  --resume <path-to-model>
```

This restores:
- model weights
- target network
- optimizer state
- episode counter
- best score

---

## 💾 Saved Models

| File                   | Meaning             |
| ---------------------- | ------------------- |
| `models/best.pt`       | Best score achieved |
| `models/last.pt`       | Last episode        |
| `models/checkpoint.pt` | Resume training     |

---

## 🎮 Play the Game

Watch AI play
```bash
python play.py \
  --map maps/map_10x10.txt \
  --N 2 \
  --wrap \
  --model models/best.pt
```

Play as human
```bash
python play.py \
  --map maps/map_10x10.txt \
  --human \
  --wrap
```

Controls:
- ⬆️ Up
- ⬇️ Down
- ⬅️ Left
- ➡️ Right

---

📺 GUI Features

- Large window
- Game grid centered
- HUD outside the grid
- Live training info:
  - episode
  - score
  - best score
  - epsilon
  - loss
  - checkpoint path

---

## 🧠 Design Decisions (Why it works well)
✔ No reverse action → cleaner action space
✔ Relative observations → translation invariant
✔ Smell vector → faster convergence
✔ Hunger shaping → prevents infinite loops
✔ Resume training → practical for long runs

---

## Advantages of NxN Sight + Smell in Snake AI

- **Generalization**: The agent learns local patterns, not fixed map layouts.
- **Scalability**: Works on any map size without changing the network.
- **Faster Learning**: Smaller state space → more stable and efficient training.
- **Robustness**: Partial observability prevents brittle, map-specific strategies.
- **Goal Awareness**: Smell (dx, dy) gives direction without solving the path.
- **Natural Behavior**: Produces snake-like movement instead of optimal but unnatural paths.
- **Transferability**: Same policy works on unseen maps and different environments.
- **Realistic Design**: Mirrors real agents (local sensors + goal direction).
- **Reduced Overfitting**: No absolute positions or full-map shortcuts.
- **Clean Action Space**: Encourages anticipation rather than memorization.
- **Small, efficient networks**: Limited vision keeps the CNN tiny, reducing compute without sacrificing performance.


---

## 🚀 Possible Extensions
- Add body-relative offsets (top-K segments)
- Add danger flags (up/down/left/right)
- LSTM for memory
- Curriculum maps (easy → hard)
- Map editor in pygame
- Imitation learning from human play

