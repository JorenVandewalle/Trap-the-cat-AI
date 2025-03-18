import pygame
import random
import heapq


# Code print voorlopig een arry af van het speelvelt als je op S klikt, dit kan gebruikt worden voor naar een AI te sturen.
# Als je op Z klikt print je alle mogelijke zetten , kan gebruikt worden voor de AI 


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

    def hexagon_points(self, x, y):
        return [(x + HEX_SIZE * 0.5, y), (x + HEX_SIZE, y + HEX_SIZE * 0.25), (x + HEX_SIZE, y + HEX_SIZE * 0.75),
                (x + HEX_SIZE * 0.5, y + HEX_SIZE), (x, y + HEX_SIZE * 0.75), (x, y + HEX_SIZE * 0.25)]
    
    def check_game_over(self):
        # Controleer of de kat op een randenpositie staat nadat hij is verplaatst
        if (self.cat_pos[0] == 0 or self.cat_pos[0] == ROWS - 1 or 
            self.cat_pos[1] == 0 or self.cat_pos[1] == COLS - 1):
            self.game_over = True
            print("Cat escaped! Game Over!")

    def move_cat(self):
        # Bepaal de kortste route naar de uitgang met A*
        path = self.a_star(self.cat_pos)
        if len(path) > 1:
            next_pos = path[1]
            # Controleer of de volgende positie geldig is en aangrenzend ligt
            if self.is_valid_move(next_pos) and self.is_adjacent(self.cat_pos, next_pos):
                print(f"Cat moving from {self.cat_pos} to {next_pos}")
                self.cat_pos = next_pos
            else:
                print(f"Invalid move attempted: {next_pos}")
        else:
            print("No valid path found for the cat!")
            self.game_over = True
            return

        # Na de zet: controleer of de kat op de rand staat
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
            if current[0] == 0 or current[0] == ROWS - 1 or current[1] == 0 or current[1] == COLS - 1:
                return path + [current]
            directions = DIRECTIONS_EVEN if current[0] % 2 == 0 else DIRECTIONS_ODD
            for d in directions:
                nr, nc = current[0] + d[0], current[1] + d[1]
                if self.is_valid_move((nr, nc)) and (nr, nc) not in visited:
                    heapq.heappush(open_set, (cost + 1 + heuristic((nr, nc)), cost + 1, (nr, nc), path + [current]))
        return []
    
    def handle_click(self, pos):
        for row in range(ROWS):
            for col in range(COLS):
                x, y = col * HEX_SIZE + (row % 2) * (HEX_SIZE // 2), row * HEX_SIZE
                hex_points = self.hexagon_points(x, y)
                if pygame.draw.polygon(self.screen, WHITE, hex_points).collidepoint(pos):
                    if (row, col) != self.cat_pos and self.grid[row][col] == 0:
                        self.grid[row][col] = 1
                        self.move_cat()
                    return
                
    def get_game_state(self):
        """Returns the current game state as a 2D array (11x11)."""
        # Copying the current grid to avoid modification outside the class
        game_state = [row[:] for row in self.grid]
        # Mark the cat position with a special value (e.g., 2) in the state array
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

    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset de game als 'R' wordt ingedrukt
                        self.reset_game()
                    elif event.key == pygame.K_s:  # Print de game state als 'S' wordt ingedrukt
                        game_state = self.get_game_state()
                        print("Game State (2D Array for AI):")
                        for row in game_state:
                            print(row)  # Print de 2D-array in de terminal
                    elif event.key == pygame.K_z:  # Print mogelijke acties als 'Z' wordt ingedrukt
                        possible_actions = self.get_possible_actions()
                        print("Mogelijke Acties (Te blokkeren vakken):")
                        for action in possible_actions:
                            print(action)  # Print de mogelijke acties in de terminal

            self.draw_grid()
            if self.game_over:
                font = pygame.font.SysFont(None, 55)
                text = font.render('Game Over! Press R to restart', True, BLACK)
                self.screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
                pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    Game().run()