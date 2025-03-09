import pygame
import random
import heapq
import numpy as np
import gym
from gym import spaces

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
            (x + HEX_SIZE * 0.5, y),  # Rechterbovenhoek
            (x + HEX_SIZE, y + HEX_SIZE * 0.25),  # Rechterkant boven
            (x + HEX_SIZE, y + HEX_SIZE * 0.75),  # Rechterkant onder
            (x + HEX_SIZE * 0.5, y + HEX_SIZE),  # Linkerbenedenhoek
            (x, y + HEX_SIZE * 0.75),  # Linkerkant onder
            (x, y + HEX_SIZE * 0.25),  # Linkerkant boven
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
        """Returns a list of all possible positions to block in the current game state."""
        actions = []
        for row in range(ROWS):
            for col in range(COLS):
                # Alleen vakken die nog niet geblokkeerd zijn (0 of 1) kunnen worden geblokkeerd
                if self.grid[row][col] == 0:  # Leeg vak of kat
                    actions.append((row, col))  # Voeg de coördinaten toe als mogelijke actie
        return actions

    
    def move_cat(self):
        if self.cat_pos[0] == 0 or self.cat_pos[0] == ROWS - 1 or self.cat_pos[1] == 0 or self.cat_pos[1] == COLS - 1:
            self.game_over = True
            print("Cat escaped! Game Over!")
            return
        
        path = self.a_star(self.cat_pos)
        if len(path) > 1:
            next_pos = path[1]
            if self.is_valid_move(next_pos) and self.is_adjacent(self.cat_pos, next_pos):
                # print(f"Cat moving from {self.cat_pos} to {next_pos}")
                self.cat_pos = next_pos
            else:
                print(f"Invalid move attempted: {next_pos}")
        else:
            print("No valid path found for the cat!")
            self.game_over = True

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
        self.observation_space = spaces.Box(low=0, high=2, shape=(ROWS, COLS), dtype=np.int8)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.game.reset_game()  # Reset de interne game status

        observation = np.array(self.game.get_game_state(), dtype=np.int8)
        #print("Game reset. Initial observation:")  # Debugging
        #print(observation)  # Debugging

        info = {}
        return observation, info

    
    def hexagon_points(self, x, y):
        return [(x + HEX_SIZE * 0.5, y), (x + HEX_SIZE, y + HEX_SIZE * 0.25), (x + HEX_SIZE, y + HEX_SIZE * 0.75),
            (x + HEX_SIZE * 0.5, y + HEX_SIZE), (x, y + HEX_SIZE * 0.75), (x, y + HEX_SIZE * 0.25)]

    
    def step(self, action):
        row, col = divmod(action, COLS)

        # Haal de mogelijke acties op
        possible_actions = self.game.get_possible_actions()
        
        if (row, col) not in possible_actions:
            print(f"Invalid action chosen by AI: ({row}, {col})")  # Debugging
            return np.array(self.game.get_game_state(), dtype=np.int8), -500, True, False, {}

        # Blokkeer de gekozen positie
        self.game.grid[row][col] = 1
        old_cat_pos = self.game.cat_pos
        self.game.move_cat()
        new_cat_pos = self.game.cat_pos

        terminated = self.game.game_over
        truncated = False
        
        # Nieuwe beloningsstrategie
        old_dist = min(old_cat_pos[0], ROWS - 1 - old_cat_pos[0], old_cat_pos[1], COLS - 1 - old_cat_pos[1])
        new_dist = min(new_cat_pos[0], ROWS - 1 - new_cat_pos[0], new_cat_pos[1], COLS - 1 - new_cat_pos[1])
        
        reward = 10  # Basisbeloning voor een geldige zet
        
        if new_dist < old_dist:
            reward -= 50  # Straf als de kat dichter bij de rand komt
        elif new_dist > old_dist:
            reward += 50  # Beloning als de kat verder van de rand blijft

        if terminated:
            if self.game.cat_pos[0] in [0, ROWS-1] or self.game.cat_pos[1] in [0, COLS-1]:
                print("Cat escaped! Game over.")
                reward = -100  # Grote straf als de kat ontsnapt
            else:
                print("Cat trapped! Game over.")
                reward = 1000  # Grote beloning als de kat vastzit

        return np.array(self.game.get_game_state(), dtype=np.int8), reward, terminated, truncated, {}





    def render(self, mode="human"):
        self.game.draw_grid()
    
    def close(self):
        pygame.quit()
