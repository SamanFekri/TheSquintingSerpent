# snake_game.py
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import pygame

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_TO_VEC = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}


def load_map_txt(path: str) -> np.ndarray:
    """
    Load a wall map from a text file.
    Each row can be:
      - "010101"
      - "0 1 0 1 0 1"
    Returns np.uint8 array (H, W), 1=wall, 0=free
    """
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if " " in line:
                row = [1 if p == "1" else 0 for p in line.split()]
            else:
                row = [1 if ch == "1" else 0 for ch in line]
            lines.append(row)

    if not lines:
        raise ValueError(f"Map file is empty: {path}")

    w = len(lines[0])
    if any(len(r) != w for r in lines):
        raise ValueError("Map rows are not the same length.")

    return np.array(lines, dtype=np.uint8)


class SnakeEnv:
    """
    Snake rules:
      - Game over if hits a wall (border if no-wrap, or wall_map=1) or bites itself.
      - wrap=True: crossing border wraps around.
    Observation returned by get_obs():
      (grid, hunger, smell)
        grid: np.float32 (1, 2N+1, 2N+1) -> encoded window around the head
              -1.0 = wall (including out-of-bounds when no-wrap)
              -0.5 = snake body (including head)
               1.0 = food
               0.0 = empty
        hunger: np.float32 scalar in [0,1]
        smell: np.float32 (2,) -> (dx, dy) normalized distance head->food (wrap-aware)
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        vision_radius: int = 1,
        max_hunger: int = 200,
        wrap: bool = True,
        wall_map: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.N = int(vision_radius)
        self.max_hunger = int(max_hunger)
        self.wrap = bool(wrap)
        self.rng = random.Random(seed)

        if wall_map is None:
            self.walls = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            wall_map = np.asarray(wall_map, dtype=np.uint8)
            if wall_map.shape != (self.height, self.width):
                raise ValueError(
                    f"wall_map shape {wall_map.shape} does not match (H,W)=({self.height},{self.width})"
                )
            self.walls = wall_map.copy()

        self.snake = deque()
        self.direction = RIGHT
        self.food: Optional[Tuple[int, int]] = None
        self.steps_since_food = 0
        self.score = 0
        self.done = False

        self.reset()

    @classmethod
    def from_map_file(
        cls,
        map_path: str,
        vision_radius: int = 1,
        max_hunger: int = 200,
        wrap: bool = True,
        seed: Optional[int] = None,
    ) -> "SnakeEnv":
        wall_map = load_map_txt(map_path)
        h, w = wall_map.shape
        return cls(
            width=w,
            height=h,
            vision_radius=vision_radius,
            max_hunger=max_hunger,
            wrap=wrap,
            wall_map=wall_map,
            seed=seed,
        )

    def _is_wall(self, x: int, y: int) -> bool:
        return self.walls[y, x] == 1

    def _is_reverse(self, action: int) -> bool:
        if self.direction == UP and action == DOWN:
            return True
        if self.direction == DOWN and action == UP:
            return True
        if self.direction == LEFT and action == RIGHT:
            return True
        if self.direction == RIGHT and action == LEFT:
            return True
        return False

    def _random_start_snake(self):
        # Try to place a 3-long horizontal snake on free cells
        for _ in range(5000):
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)

            if self.wrap:
                cand = [(x, y), ((x - 1) % self.width, y), ((x - 2) % self.width, y)]
            else:
                cand = [(x, y), (x - 1, y), (x - 2, y)]
                if any(xx < 0 or xx >= self.width for (xx, yy) in cand):
                    continue

            if len(set(cand)) != 3:
                continue
            if any(self._is_wall(xx, yy) for (xx, yy) in cand):
                continue

            return cand

        # fallback
        cx, cy = self.width // 2, self.height // 2
        return [(cx, cy), (max(cx - 1, 0), cy), (max(cx - 2, 0), cy)]

    def _spawn_food(self):
        snake_set = set(self.snake)
        empty = []
        for y in range(self.height):
            for x in range(self.width):
                if self.walls[y, x] == 0 and (x, y) not in snake_set:
                    empty.append((x, y))
        if not empty:
            self.food = None
            self.done = True
            return
        self.food = self.rng.choice(empty)

    def reset(self):
        self.done = False
        self.score = 0
        self.steps_since_food = 0

        # prefer center start, else random safe start
        cx, cy = self.width // 2, self.height // 2
        init = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        if not self.wrap and any(xx < 0 for (xx, yy) in init):
            init = self._random_start_snake()
        else:
            if self.wrap:
                init = [(cx % self.width, cy % self.height),
                        ((cx - 1) % self.width, cy % self.height),
                        ((cx - 2) % self.width, cy % self.height)]
            if any(self._is_wall(x, y) for (x, y) in init):
                init = self._random_start_snake()

        self.snake = deque(init)
        self.direction = RIGHT
        self._spawn_food()
        return self.get_obs()

    @staticmethod
    def _shortest_wrap_delta(delta: int, size: int) -> int:
        # Convert delta into shortest signed delta under wrap
        half = size // 2
        if delta > half:
            delta -= size
        elif delta < -half:
            delta += size
        return delta

    def step(self, action: int):
        if self.done:
            return self.get_obs(), 0.0, True, {"reason": "done"}

        if self._is_reverse(action):
            action = self.direction

        self.direction = action
        dx, dy = ACTION_TO_VEC[action]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        # border handling
        if self.wrap:
            nx %= self.width
            ny %= self.height
        else:
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                self.done = True
                return self.get_obs(), -1.0, True, {"reason": "wall"}

        # wall map collision
        if self._is_wall(nx, ny):
            self.done = True
            return self.get_obs(), -1.0, True, {"reason": "wall"}

        # self collision
        if (nx, ny) in self.snake:
            self.done = True
            return self.get_obs(), -1.0, True, {"reason": "self"}

        reward = 0.0
        self.snake.appendleft((nx, ny))

        # eat?
        if self.food is not None and (nx, ny) == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward += 1.0
            self._spawn_food()
            if self.done and self.food is None:
                return self.get_obs(), reward, True, {"reason": "win"}
        else:
            self.snake.pop()
            self.steps_since_food += 1

            hunger = min(self.steps_since_food / self.max_hunger, 1.0)
            reward += -0.01 - 0.02 * hunger

        return self.get_obs(), reward, self.done, {"reason": ""}

    def get_obs(self):
        # local grid encoded in a single channel
        size = 2 * self.N + 1
        grid_2d = np.zeros((size, size), dtype=np.float32)

        hx, hy = self.snake[0]
        snake_set = set(self.snake)

        for j in range(size):
            for i in range(size):
                x = hx + (i - self.N)
                y = hy + (j - self.N)

                if self.wrap:
                    x %= self.width
                    y %= self.height
                else:
                    if x < 0 or x >= self.width or y < 0 or y >= self.height:
                        grid_2d[j, i] = -1.0
                        continue

                if self.walls[y, x] == 1:
                    grid_2d[j, i] = -1.0
                elif (x, y) in snake_set:
                    grid_2d[j, i] = -0.5
                elif self.food is not None and (x, y) == self.food:
                    grid_2d[j, i] = 1.0

        grid = np.expand_dims(grid_2d, axis=0)  # (1,H,W)
        hunger = np.float32(min(self.steps_since_food / self.max_hunger, 1.0))

        # smell = normalized (dx, dy) from head to food (wrap-aware)
        if self.food is None:
            smell = np.array([0.0, 0.0], dtype=np.float32)
        else:
            fx, fy = self.food
            dx = fx - hx
            dy = fy - hy
            if self.wrap:
                dx = self._shortest_wrap_delta(dx, self.width)
                dy = self._shortest_wrap_delta(dy, self.height)
            smell = np.array([dx / self.width, dy / self.height], dtype=np.float32)

        return grid, hunger, smell


class SnakeRenderer:
    def __init__(
        self,
        env: SnakeEnv,
        cell_size: int = 24,
        fps: int = 15,
        padding: int = 80,   # 👈 space around the game
    ):
        self.env = env
        self.cell = cell_size
        self.fps = fps
        self.padding = padding

        self.grid_w = env.width * cell_size
        self.grid_h = env.height * cell_size

        self.win_w = self.grid_w + padding * 2
        self.win_h = self.grid_h + padding * 2

        # top-left corner where the grid starts (centered)
        self.offset_x = (self.win_w - self.grid_w) // 2
        self.offset_y = (self.win_h - self.grid_h) // 2

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Snake (AI / Human)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        self.status_lines = []

    def set_status(self, lines):
        self.status_lines = list(lines)

    def close(self):
        pygame.quit()

    def poll_quit(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def get_human_action(self, current_dir: int) -> int:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return UP
        if keys[pygame.K_DOWN]:
            return DOWN
        if keys[pygame.K_LEFT]:
            return LEFT
        if keys[pygame.K_RIGHT]:
            return RIGHT
        return current_dir

    def _cell_rect(self, x, y):
        return pygame.Rect(
            self.offset_x + x * self.cell,
            self.offset_y + y * self.cell,
            self.cell,
            self.cell,
        )

    def draw(self):
        env = self.env
        self.screen.fill((15, 15, 15))  # dark background

        # --- grid background (optional visual frame) ---
        pygame.draw.rect(
            self.screen,
            (30, 30, 30),
            pygame.Rect(
                self.offset_x - 4,
                self.offset_y - 4,
                self.grid_w + 8,
                self.grid_h + 8,
            ),
            border_radius=8,
        )

        # --- walls ---
        for y in range(env.height):
            for x in range(env.width):
                if env.walls[y, x] == 1:
                    pygame.draw.rect(
                        self.screen,
                        (90, 90, 90),
                        self._cell_rect(x, y),
                        border_radius=4,
                    )

        # --- food ---
        if env.food is not None:
            fx, fy = env.food
            pygame.draw.rect(
                self.screen,
                (220, 60, 60),
                self._cell_rect(fx, fy),
                border_radius=6,
            )

        # --- snake ---
        for idx, (x, y) in enumerate(env.snake):
            color = (70, 220, 130) if idx == 0 else (50, 170, 105)
            pygame.draw.rect(
                self.screen,
                color,
                self._cell_rect(x, y),
                border_radius=6,
            )

        # --- HUD (outside the grid) ---
        hunger = min(env.steps_since_food / env.max_hunger, 1.0)
        hud = [f"Score: {env.score}   Hunger: {hunger:.2f}"] + self.status_lines

        y0 = 10
        for line in hud[:8]:
            surf = self.font.render(line, True, (230, 230, 230))
            self.screen.blit(surf, (10, y0))
            y0 += 20

        pygame.display.flip()
        self.clock.tick(self.fps)
