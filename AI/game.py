import pygame
import random
import heapq
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 11, 11
HEX_SIZE = WIDTH // (COLS + 1)
WHITE, BLACK, GRAY, BLUE = (255, 255, 255), (0, 0, 0), (200, 200, 200), (50, 50, 255)

# Directions for hexagonal movement based on even/odd rows
DIRECTIONS_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
DIRECTIONS_ODD = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Trap the Cat")
        self.reset_game()
    
    def reset_game(self):
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.cat_pos = (ROWS // 2, COLS // 2)
        self.running = True
        self.game_over = False
        self.trapped = False
        self.init_blocks()

    def init_blocks(self):
        for _ in range(20):
            while True:
                row, col = random.randint(0, ROWS - 1), random.randint(0, COLS - 1)
                if (row, col) != self.cat_pos and self.grid[row][col] == 0:
                    self.grid[row][col] = 1
                    break


    def hexagon_points(self, x, y):
        """Berekent de hoekpunten van een hexagoon op positie (x, y)."""
        return [
            (x + HEX_SIZE * 0.5, y),
            (x + HEX_SIZE, y + HEX_SIZE * 0.25),
            (x + HEX_SIZE, y + HEX_SIZE * 0.75),
            (x + HEX_SIZE * 0.5, y + HEX_SIZE),
            (x, y + HEX_SIZE * 0.75),
            (x, y + HEX_SIZE * 0.25),
        ]

    def draw_grid(self):
        self.screen.fill(WHITE)
        for row in range(ROWS):
            for col in range(COLS):
                x, y = col * HEX_SIZE + (row % 2) * (HEX_SIZE // 2), row * HEX_SIZE
                color = GRAY if self.grid[row][col] == 1 else WHITE
                pygame.draw.polygon(self.screen, color, self.hexagon_points(x, y))
                pygame.draw.polygon(self.screen, BLACK, self.hexagon_points(x, y), 1)
        
        # Draw the cat
        x, y = self.cat_pos[1] * HEX_SIZE + (self.cat_pos[0] % 2) * (HEX_SIZE // 2), self.cat_pos[0] * HEX_SIZE
        pygame.draw.circle(self.screen, BLUE, (x + HEX_SIZE // 2, y + HEX_SIZE // 2), HEX_SIZE // 3)
        pygame.display.flip()

    def get_game_state(self):
        """Returns the current game state as a 2D array (11x11)."""
        game_state = [row[:] for row in self.grid]
        game_state[self.cat_pos[0]][self.cat_pos[1]] = 2
        return game_state

    def get_possible_actions(self):
        """Geeft alleen niet-geblokkeerde cellen terug, exclusief katpositie"""
        return [
            (row, col) 
            for row in range(ROWS) 
            for col in range(COLS) 
            if self.grid[row][col] == 0 and (row, col) != self.cat_pos
        ]

    def check_game_over(self):
        # Als de kat op de rand staat, ontsnapt hij
        if (self.cat_pos[0] == 0 or self.cat_pos[0] == ROWS - 1 or 
            self.cat_pos[1] == 0 or self.cat_pos[1] == COLS - 1):
            self.game_over = True
            #print("Cat escaped! Game Over!")
        # Als er geen geldige zetten meer zijn (en dus geen pad), is de kat gevangen
        elif len(self.a_star(self.cat_pos)) == 0:
            self.game_over = True
            self.trapped = True
            print("Cat trapped! Game Over!")

    def move_cat(self):
        # Bepaal de kortste route naar de rand met A*
        path = self.a_star(self.cat_pos)
        if len(path) > 1:
            next_pos = path[1]
            # Controleer of de volgende positie geldig is en aangrenzend ligt
            if self.is_valid_move(next_pos) and self.is_adjacent(self.cat_pos, next_pos):
                #print(f"Cat moving from {self.cat_pos} to {next_pos}")
                self.cat_pos = next_pos
            else:
                print(f"Invalid move attempted: {next_pos}")
        else:
            print("No valid path found for the cat!")
            self.game_over = True
            self.trapped = True
            return

        # Na de zet: controleer of de kat op de rand staat of gevangen zit
        self.check_game_over()

    def is_valid_move(self, pos):
        row, col = pos
        return 0 <= row < ROWS and 0 <= col < COLS and self.grid[row][col] == 0

    def is_adjacent(self, pos1, pos2):
        row, col = pos1
        directions = DIRECTIONS_EVEN if row % 2 == 0 else DIRECTIONS_ODD
        return pos2 in [(row + d[0], col + d[1]) for d in directions]

    def a_star(self, start):
        def heuristic(pos):
            return min(pos[0], ROWS - 1 - pos[0], pos[1], COLS - 1 - pos[1])
        
        open_set = [(heuristic(start), 0, start, [])]
        visited = set()
        while open_set:
            _, cost, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            # Als current op de rand ligt: we hebben een pad gevonden
            if current[0] == 0 or current[0] == ROWS - 1 or current[1] == 0 or current[1] == COLS - 1:
                return path + [current]
            directions = DIRECTIONS_EVEN if current[0] % 2 == 0 else DIRECTIONS_ODD
            for d in directions:
                nr, nc = current[0] + d[0], current[1] + d[1]
                if self.is_valid_move((nr, nc)) and (nr, nc) not in visited:
                    heapq.heappush(open_set, (cost + 1 + heuristic((nr, nc)), cost + 1, (nr, nc), path + [current]))
        return []

class TrapTheCatEnv(gym.Env):
    def __init__(self, game):
        super(TrapTheCatEnv, self).__init__()
        self.game = game
        self.action_space = spaces.Discrete(ROWS * COLS)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=2, shape=(ROWS, COLS)), 
            "cat_pos": spaces.Box(low=0, high=10, shape=(2,), dtype=np.int8)
        })

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        self.game.reset_game()
        observation = {
            "grid": np.array(self.game.grid, dtype=np.int8),
            "cat_pos": np.array(self.game.cat_pos, dtype=np.int8)
        }
        return observation, {}

    def hexagon_points(self, x, y):
        return [
            (x + HEX_SIZE * 0.5, y),
            (x + HEX_SIZE, y + HEX_SIZE * 0.25),
            (x + HEX_SIZE, y + HEX_SIZE * 0.75),
            (x + HEX_SIZE * 0.5, y + HEX_SIZE),
            (x, y + HEX_SIZE * 0.75),
            (x, y + HEX_SIZE * 0.25)
        ]
    
    def step(self, action):
        row, col = divmod(action, COLS)
        possible_actions = self.game.get_possible_actions()
        terminated = False
        truncated = False
        info = {}

        if (row, col) not in possible_actions:
            return (
                {"grid": np.array(self.game.grid), "cat_pos": np.array(self.game.cat_pos)},
                -1000,
                False,
                False,
                {}
            )

        self.game.grid[row][col] = 1
        old_pos = self.game.cat_pos
        old_dist = min(old_pos[0], ROWS-1 - old_pos[0], old_pos[1], COLS-1 - old_pos[1])
        self.game.move_cat()
        new_pos = self.game.cat_pos
        new_dist = min(new_pos[0], ROWS-1 - new_pos[0], new_pos[1], COLS-1 - new_pos[1])

        reward = 0
        if new_dist < old_dist:
            reward -= 100 + (old_dist - new_dist) * 20
        elif new_dist > old_dist:
            reward += 50 + (new_dist - old_dist) * 10

        if self.game.game_over:
            terminated = True
            if self.game.trapped:
                reward += 5000
            else:
                reward -= 2000

        info = {
            "cat_pos": new_pos,
            "old_dist": old_dist,
            "new_dist": new_dist
        }

        return (
            {"grid": np.array(self.game.grid), "cat_pos": np.array(self.game.cat_pos)},
            reward,
            terminated,
            truncated,
            info
        )

    def render(self, mode="human"):
        self.game.draw_grid()
    
    def close(self):
        pygame.quit()
