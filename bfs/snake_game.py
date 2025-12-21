# snake_game.py
import random
from collections import deque
from typing import Optional, Tuple, List

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
    Supported per row:
      - "010010"
      - "0 1 0 0 1 0"
    Returns: np.uint8 array (H,W), 1=wall 0=free
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if " " in line:
                parts = line.split()
                row = [1 if p == "1" else 0 for p in parts]
            else:
                row = [1 if ch == "1" else 0 for ch in line]
            rows.append(row)

    if not rows:
        raise ValueError(f"Empty map file: {path}")

    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("Map rows have different lengths")

    return np.array(rows, dtype=np.uint8)


class SnakeEnv:
    """
    Snake environment:
      - Game over if hits wall or bites itself.
      - wrap=True makes borders wrap-around.
      - Start length configurable (default 1).

    Observation given to agent:
      (grid, hunger, smell, lidar)
        grid  : float32 (3, 2N+1, 2N+1) [walls, body, food]
        hunger: float32 scalar in [0,1]
        smell : float32 (2,) normalized (dx,dy) head->food (wrap-aware shortest)
        lidar : float32 (2*num_rays,) => [dist_to_body rays..., dist_to_wall rays...]
                Each dist is normalized to [0,1], where 1 means "not seen within max_range".

    BFS is used ONLY for reward shaping inside env (agent does NOT see BFS results).
    """

    def __init__(
        self,
        width: int,
        height: int,
        vision_radius: int = 2,
        max_hunger: int = 200,
        hunger_step: float = 1.0,
        wrap: bool = True,
        wall_map: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        num_rays: int = 16,
        start_length: int = 1,
        bfs_shaping: bool = True,
        bfs_space_penalty: float = 0.05,
        bfs_tail_penalty: float = 0.05,
        death_penalty: float = 100.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.N = int(vision_radius)
        self.max_hunger = int(max_hunger)
        self.hunger_step = float(hunger_step)
        self.wrap = bool(wrap)
        self.num_rays = int(num_rays)
        self.start_length = max(1, int(start_length))

        self.bfs_shaping = bool(bfs_shaping)
        self.bfs_space_penalty = float(bfs_space_penalty)
        self.bfs_tail_penalty = float(bfs_tail_penalty)
        self.death_penalty = float(death_penalty)

        self.rng = random.Random(seed)

        if wall_map is None:
            self.walls = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            wall_map = np.asarray(wall_map, dtype=np.uint8)
            if wall_map.shape != (self.height, self.width):
                raise ValueError(
                    f"wall_map shape {wall_map.shape} != ({self.height},{self.width})"
                )
            self.walls = wall_map.copy()

        self.snake = deque()
        self.direction = RIGHT
        self.food: Optional[Tuple[int, int]] = None
        self.steps_since_food = 0
        self.score = 0
        self.done = False

        self.reset()

    def use_global_random(self):
        """Use the global ``random`` module for environment randomness.

        This keeps food placement and other stochastic events aligned with
        the global seed configured by callers.
        """
        self.rng = random

    @classmethod
    def from_map_file(
        cls,
        map_path: str,
        vision_radius: int = 2,
        max_hunger: int = 200,
        hunger_step: float = 1.0,
        wrap: bool = True,
        seed: Optional[int] = None,
        num_rays: int = 16,
        start_length: int = 1,
        bfs_shaping: bool = True,
        bfs_space_penalty: float = 0.5,
        bfs_tail_penalty: float = 0.5,
        death_penalty: float = 5.0,
    ) -> "SnakeEnv":
        wall_map = load_map_txt(map_path)
        h, w = wall_map.shape
        return cls(
            width=w,
            height=h,
            vision_radius=vision_radius,
            max_hunger=max_hunger,
            hunger_step=hunger_step,
            wrap=wrap,
            wall_map=wall_map,
            seed=seed,
            num_rays=num_rays,
            start_length=start_length,
            bfs_shaping=bfs_shaping,
            bfs_space_penalty=bfs_space_penalty,
            bfs_tail_penalty=bfs_tail_penalty,
            death_penalty=death_penalty,
        )

    def _is_wall(self, x: int, y: int) -> bool:
        return self.walls[y, x] == 1

    def _is_reverse(self, action: int) -> bool:
        return (
            (self.direction == UP and action == DOWN)
            or (self.direction == DOWN and action == UP)
            or (self.direction == LEFT and action == RIGHT)
            or (self.direction == RIGHT and action == LEFT)
        )

    @staticmethod
    def _shortest_wrap_delta(delta: int, size: int) -> int:
        half = size // 2
        if delta > half:
            delta -= size
        elif delta < -half:
            delta += size
        return delta

    def _random_start(self):
        # pick a random free cell for the head
        for _ in range(10000):
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)
            if not self._is_wall(x, y):
                return (x, y)
        return (self.width // 2, self.height // 2)

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

        hx, hy = self._random_start()
        self.snake = deque([(hx, hy)])  # ✅ start_length=1 behavior

        # If you want start_length > 1, extend behind to the left (wrap-aware) on free cells
        # This will try a few directions; if fails, it stays length 1.
        if self.start_length > 1:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            placed = [(hx, hy)]
            for dx, dy in directions:
                placed = [(hx, hy)]
                ok = True
                for k in range(1, self.start_length):
                    x = hx - dx * k
                    y = hy - dy * k
                    if self.wrap:
                        x %= self.width
                        y %= self.height
                    else:
                        if x < 0 or x >= self.width or y < 0 or y >= self.height:
                            ok = False
                            break
                    if self._is_wall(x, y):
                        ok = False
                        break
                    if (x, y) in placed:
                        ok = False
                        break
                    placed.append((x, y))
                if ok:
                    break
            self.snake = deque(placed)

        self.direction = RIGHT
        self._spawn_food()
        return self.get_obs()

    # ---------------- BFS (BODY-AWARE) ----------------

    def bfs_reachable_free_space(self, start, *, include_tail_as_free: bool = True) -> int:
        """
        BFS count reachable cells from start.

        Obstacles:
          - walls blocked
          - body blocked
          - tail optionally treated as free (good when tail will move)
        """
        sx, sy = start
        visited = set([(sx, sy)])
        q = deque([(sx, sy)])

        body = set(self.snake)
        tail = self.snake[-1] if len(self.snake) > 0 else None

        count = 0
        while q:
            x, y = q.popleft()
            count += 1

            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy

                if self.wrap:
                    nx %= self.width
                    ny %= self.height
                else:
                    if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                        continue

                if (nx, ny) in visited:
                    continue

                if self.walls[ny, nx] == 1:
                    continue

                if (nx, ny) in body:
                    if not include_tail_as_free:
                        continue
                    if tail is None or (nx, ny) != tail:
                        continue

                visited.add((nx, ny))
                q.append((nx, ny))

        return count

    def bfs_can_reach_tail(self, *, include_tail_as_free: bool = True) -> bool:
        """
        BFS: can head reach tail?
        Body blocks except tail (tail allowed).
        """
        if len(self.snake) < 2:
            return True

        hx, hy = self.snake[0]
        tx, ty = self.snake[-1]
        tail = (tx, ty)

        visited = set([(hx, hy)])
        q = deque([(hx, hy)])
        body = set(self.snake)

        while q:
            x, y = q.popleft()
            if (x, y) == tail:
                return True

            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy

                if self.wrap:
                    nx %= self.width
                    ny %= self.height
                else:
                    if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                        continue

                if (nx, ny) in visited:
                    continue

                if self.walls[ny, nx] == 1:
                    continue

                if (nx, ny) in body:
                    if not include_tail_as_free:
                        continue
                    if (nx, ny) != tail:
                        continue

                visited.add((nx, ny))
                q.append((nx, ny))

        return False

    # ---------------- LIDAR ----------------

    def _ray_dirs(self) -> List[Tuple[int, int]]:
        # 16 rays: 8 cardinal/diagonal + 8 extra slopes
        dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
            (2, -1), (2, 1), (1, 2), (-1, 2),
            (-2, 1), (-2, -1), (-1, -2), (1, -2),
        ]
        return dirs[: self.num_rays]

    def body_wall_lidar(self, max_range: Optional[int] = None) -> np.ndarray:
        """
        Returns float32 vector length 2*num_rays:
          [dist_to_body..., dist_to_wall...]
        Distances normalized by max_range. 1.0 means not found within range.
        """
        if max_range is None:
            max_range = self.width + self.height
        max_range = int(max_range)

        hx, hy = self.snake[0]
        body_set = set(list(self.snake)[1:])  # exclude head
        dirs = self._ray_dirs()

        body_out = np.ones((len(dirs),), dtype=np.float32)
        wall_out = np.ones((len(dirs),), dtype=np.float32)

        for i, (dx, dy) in enumerate(dirs):
            x, y = hx, hy
            body_hit = None

            for step in range(1, max_range + 1):
                x += dx
                y += dy

                if self.wrap:
                    x %= self.width
                    y %= self.height
                else:
                    if x < 0 or x >= self.width or y < 0 or y >= self.height:
                        wall_out[i] = step / max_range
                        break

                if self.walls[y, x] == 1:
                    wall_out[i] = step / max_range
                    break

                if body_hit is None and (x, y) in body_set:
                    body_hit = step

            if body_hit is not None:
                body_out[i] = body_hit / max_range

        return np.concatenate([body_out, wall_out], axis=0).astype(np.float32)

    # ---------------- OBS ----------------

    def get_obs(self):
        size = 2 * self.N + 1
        walls_ch = np.zeros((size, size), dtype=np.float32)
        body_ch = np.zeros((size, size), dtype=np.float32)
        food_ch = np.zeros((size, size), dtype=np.float32)

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
                        walls_ch[j, i] = 1.0
                        continue

                if self.walls[y, x] == 1:
                    walls_ch[j, i] = 1.0
                if (x, y) in snake_set:
                    body_ch[j, i] = 1.0
                if self.food is not None and (x, y) == self.food:
                    food_ch[j, i] = 1.0

        grid = np.stack([walls_ch, body_ch, food_ch], axis=0)
        hunger = np.float32(self.steps_since_food / self.max_hunger)

        # smell vector
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

        lidar = self.body_wall_lidar()  # (2*num_rays,)

        return grid, hunger, smell, lidar

    # ---------------- STEP (with BFS shaping) ----------------

    def step(self, action: int):
        if self.done:
            return self.get_obs(), 0.0, True, {"reason": "done"}

        if self._is_reverse(action):
            action = self.direction

        self.direction = action
        dx, dy = ACTION_TO_VEC[action]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        # border
        if self.wrap:
            nx %= self.width
            ny %= self.height
        else:
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                self.done = True
                return self.get_obs(), -self.death_penalty, True, {"reason": "wall"}

        # wall
        if self._is_wall(nx, ny):
            self.done = True
            return self.get_obs(), -self.death_penalty, True, {"reason": "wall"}

        # self collision
        if (nx, ny) in self.snake:
            self.done = True
            return self.get_obs(), -self.death_penalty, True, {"reason": "self"}

        reward = 0.0
        self.snake.appendleft((nx, ny))

        ate = (self.food is not None and (nx, ny) == self.food)

        if ate:
            self.score += 1
            self.steps_since_food = 0
            reward += 1.0
            self._spawn_food()
        else:
            self.snake.pop()
            self.steps_since_food += self.hunger_step
            hunger = self.steps_since_food / self.max_hunger
            reward += -0.01 - 0.02 * hunger

        # ✅ BFS reward shaping (environment-only)
        if self.bfs_shaping and not self.done:
            # Tail is only "free" if we did NOT eat (because tail moves only then)
            include_tail = (not ate)

            free_space = self.bfs_reachable_free_space(self.snake[0], include_tail_as_free=include_tail)
            if free_space < len(self.snake):
                reward -= self.bfs_space_penalty

            if not self.bfs_can_reach_tail(include_tail_as_free=include_tail):
                reward -= self.bfs_tail_penalty

        return self.get_obs(), reward, self.done, {"reason": ""}


class SnakeRenderer:
    """Big window, centered grid, HUD outside grid."""
    def __init__(self, env: SnakeEnv, cell_size: int = 24, fps: int = 15, padding: int = 90):
        self.env = env
        self.cell = cell_size
        self.fps = fps
        self.padding = padding

        self.grid_w = env.width * cell_size
        self.grid_h = env.height * cell_size
        self.win_w = self.grid_w + padding * 2
        self.win_h = self.grid_h + padding * 2
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
        self.screen.fill((15, 15, 15))

        # frame around grid
        pygame.draw.rect(
            self.screen,
            (30, 30, 30),
            pygame.Rect(self.offset_x - 4, self.offset_y - 4, self.grid_w + 8, self.grid_h + 8),
            border_radius=10,
        )

        # walls
        for y in range(env.height):
            for x in range(env.width):
                if env.walls[y, x] == 1:
                    pygame.draw.rect(self.screen, (90, 90, 90), self._cell_rect(x, y), border_radius=4)

        # food
        if env.food is not None:
            fx, fy = env.food
            pygame.draw.rect(self.screen, (220, 60, 60), self._cell_rect(fx, fy), border_radius=6)

        # snake
        for idx, (x, y) in enumerate(env.snake):
            color = (70, 220, 130) if idx == 0 else (50, 170, 105)
            pygame.draw.rect(self.screen, color, self._cell_rect(x, y), border_radius=6)

        hunger = env.steps_since_food / env.max_hunger
        hud = [f"Score: {env.score}   Hunger: {hunger:.2f}   Wrap: {env.wrap}"] + self.status_lines

        y0 = 10
        for line in hud[:12]:
            surf = self.font.render(line, True, (230, 230, 230))
            self.screen.blit(surf, (10, y0))
            y0 += 20

        pygame.display.flip()
        self.clock.tick(self.fps)
