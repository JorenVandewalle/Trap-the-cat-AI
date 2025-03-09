import pygame
import random
import heapq

# Constants voor het venster
WIDTH, HEIGHT = 1200, 1000
ROWS, COLS = 11, 11

# Kleur definities
WHITE, BLACK, GRAY, BLUE = (255, 255, 255), (0, 0, 0), (200, 200, 200), (50, 50, 255)

# Richtingen voor hexagonale beweging (afhankelijk van even/oneven rijen)
DIRECTIONS_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
DIRECTIONS_ODD  = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Trap the Cat")
        
        # Stel de linker zijbalk in (voor instructies) en bereken de beschikbare ruimte voor het grid
        self.left_sidebar_width = 300
        self.game_area_width = WIDTH - self.left_sidebar_width
        self.hex_size = self.game_area_width // (COLS + 1)
        
        # Load the cat image and scale it to 66% of the hex cell size
        self.cat_image = pygame.image.load("images/cat_icon4.png").convert_alpha()
        scale = int(self.hex_size * 0.66)
        self.cat_image = pygame.transform.scale(self.cat_image, (scale, scale))
        
        self.num_blocks = 10  # Standaard aantal blokken
        self.running = True
        self.reset_game()

    def reset_game(self):
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.cat_pos = (ROWS // 2, COLS // 2)
        self.game_over = False
        self.cat_trapped = False  # Houdt bij of de kat al vastzit
        self.first_move = False   # Geeft aan of er al een zet is gedaan
        self.win = None           # True als gewonnen, False als verloren
        self.init_blocks()

    def init_blocks(self):
        # Plaats de vooraf ingestelde blokken
        for _ in range(self.num_blocks):
            while True:
                row, col = random.randint(0, ROWS - 1), random.randint(0, COLS - 1)
                if (row, col) != self.cat_pos and self.grid[row][col] == 0:
                    self.grid[row][col] = 1
                    break

    def get_grid_offsets(self):
        """
        Berekent de offsets zodat het grid gecentreerd staat binnen het game-gebied (rechts van de zijbalk).
        """
        grid_draw_width = COLS * self.hex_size + self.hex_size // 2
        grid_draw_height = ROWS * self.hex_size
        grid_offset_x = self.left_sidebar_width + (self.game_area_width - grid_draw_width) // 2
        grid_offset_y = (HEIGHT - grid_draw_height) // 2
        return grid_offset_x, grid_offset_y

    def hexagon_points(self, x, y):
        return [
            (x + self.hex_size * 0.5, y),
            (x + self.hex_size, y + self.hex_size * 0.25),
            (x + self.hex_size, y + self.hex_size * 0.75),
            (x + self.hex_size * 0.5, y + self.hex_size),
            (x, y + self.hex_size * 0.75),
            (x, y + self.hex_size * 0.25)
        ]
    
    def draw_instructions(self):
        """Tekent de instructies/controls in de linker zijbalk."""
        font = pygame.font.SysFont(None, 24)
        instructions = [
            "Controls:",
            "Click: plaats blok",
            "R: reset spel",
            "1: 10 blokken",
            "2: 20 blokken",
            "3: 30 blokken",
            "",
            "Goal:",
            "Trap de Kat!",
            "",
            f"Huidig aantal blokken: {self.num_blocks}"
        ]
        x = 10
        y = 10
        for line in instructions:
            text = font.render(line, True, BLACK)
            self.screen.blit(text, (x, y))
            y += text.get_height() + 5

    def draw_grid(self):
        # Vul het scherm met wit
        self.screen.fill(WHITE)
        # Teken instructies in de linker zijbalk
        self.draw_instructions()
        
        grid_offset_x, grid_offset_y = self.get_grid_offsets()
        
        # Teken het grid in het rechtergedeelte
        for row in range(ROWS):
            for col in range(COLS):
                x = grid_offset_x + col * self.hex_size + (row % 2) * (self.hex_size // 2)
                y = grid_offset_y + row * self.hex_size
                color = GRAY if self.grid[row][col] == 1 else WHITE
                pygame.draw.polygon(self.screen, color, self.hexagon_points(x, y))
                pygame.draw.polygon(self.screen, BLACK, self.hexagon_points(x, y), 1)
        
        # Teken de kat met het geladen plaatje
        cat_x = grid_offset_x + self.cat_pos[1] * self.hex_size + (self.cat_pos[0] % 2) * (self.hex_size // 2)
        cat_y = grid_offset_y + self.cat_pos[0] * self.hex_size
        image_rect = self.cat_image.get_rect(center=(cat_x + self.hex_size // 2, cat_y + self.hex_size // 2))
        self.screen.blit(self.cat_image, image_rect)

    def move_cat(self):
        # Als de kat aan de rand staat, ontsnapt hij en is het spel verloren
        if (self.cat_pos[0] == 0 or self.cat_pos[0] == ROWS - 1 or
            self.cat_pos[1] == 0 or self.cat_pos[1] == COLS - 1):
            self.game_over = True
            self.win = False
            print("Cat escaped! Game Over!")
            return
        
        # Gebruik A* om het beste pad te vinden als de kat nog niet vastzit
        if not self.cat_trapped:
            path = self.a_star(self.cat_pos)
            if len(path) > 1:
                next_pos = path[1]
                if self.is_valid_move(next_pos) and self.is_adjacent(self.cat_pos, next_pos):
                    print(f"Cat moving from {self.cat_pos} to {next_pos}")
                    self.cat_pos = next_pos
                else:
                    print(f"Invalid move attempted: {next_pos}")
            else:
                print("Cat is trapped! Switching to random moves.")
                self.cat_trapped = True
        
        # Als de kat vastzit, maak een willekeurige geldige zet
        if self.cat_trapped:
            valid_moves = self.get_valid_moves(self.cat_pos)
            if valid_moves:
                next_pos = random.choice(valid_moves)
                print(f"Cat moving randomly from {self.cat_pos} to {next_pos}")
                self.cat_pos = next_pos
            else:
                self.game_over = True
                self.win = True
                print("Cat is completely trapped! Game Over!")

    def get_valid_moves(self, pos):
        row, col = pos
        directions = DIRECTIONS_EVEN if row % 2 == 0 else DIRECTIONS_ODD
        valid_moves = []
        for d in directions:
            nr, nc = row + d[0], col + d[1]
            if self.is_valid_move((nr, nc)):
                valid_moves.append((nr, nc))
        return valid_moves
    
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
            if (current[0] == 0 or current[0] == ROWS - 1 or
                current[1] == 0 or current[1] == COLS - 1):
                return path + [current]
            directions = DIRECTIONS_EVEN if current[0] % 2 == 0 else DIRECTIONS_ODD
            for d in directions:
                nr, nc = current[0] + d[0], current[1] + d[1]
                if self.is_valid_move((nr, nc)) and (nr, nc) not in visited:
                    heapq.heappush(open_set, (cost + 1 + heuristic((nr, nc)),
                                               cost + 1, (nr, nc), path + [current]))
        return []
    
    def handle_click(self, pos):
        grid_offset_x, grid_offset_y = self.get_grid_offsets()
        for row in range(ROWS):
            for col in range(COLS):
                x = grid_offset_x + col * self.hex_size + (row % 2) * (self.hex_size // 2)
                y = grid_offset_y + row * self.hex_size
                hex_points = self.hexagon_points(x, y)
                if pygame.draw.polygon(self.screen, WHITE, hex_points).collidepoint(pos):
                    if (row, col) != self.cat_pos and self.grid[row][col] == 0:
                        self.grid[row][col] = 1
                        self.first_move = True  # Een zet is geplaatst
                        self.move_cat()
                    return

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    # Reset het spel met 'R'
                    if event.key == pygame.K_r:
                        self.reset_game()
                    # Wijzig het aantal blokken zolang er nog geen zet is gedaan
                    elif not self.first_move:
                        if event.key == pygame.K_1:
                            self.num_blocks = 10
                            print("Aantal blokken ingesteld op 10")
                            self.reset_game()
                        elif event.key == pygame.K_2:
                            self.num_blocks = 20
                            print("Aantal blokken ingesteld op 20")
                            self.reset_game()
                        elif event.key == pygame.K_3:
                            self.num_blocks = 30
                            print("Aantal blokken ingesteld op 30")
                            self.reset_game()
            
            self.draw_grid()
            # Als het spel voorbij is, toon het juiste bericht
            if self.game_over:
                font = pygame.font.SysFont(None, 55)
                if self.win:
                    message = 'Je hebt gewonnen! De kat is gevangen. Druk op R om te resetten'
                else:
                    message = 'Je hebt verloren! De kat is ontsnapt. Druk op R om te resetten'
                text = font.render(message, True, BLACK)
                self.screen.blit(text, ((WIDTH - text.get_width()) // 2,
                                        (HEIGHT - text.get_height()) // 2))
            pygame.display.flip()
            clock.tick(30)  # Limiteer tot 30 FPS
        pygame.quit()

if __name__ == "__main__":
    Game().run()
