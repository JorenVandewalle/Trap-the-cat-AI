import pygame
import random
import heapq

# Constants voor het venster en kleuren
WIDTH, HEIGHT = 1200, 1000
WHITE, BLACK, GRAY, BLUE = (255, 255, 255), (0, 0, 0), (200, 200, 200), (50, 50, 255)

# Richtingen voor hexagonale beweging (afhankelijk van even/oneven rijen)
DIRECTIONS_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
DIRECTIONS_ODD  = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

class Game:
    def __init__(self, ai_mode=False):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        caption = "Trap the Cat - AI Block Placer" if ai_mode else "Trap the Cat"
        pygame.display.set_caption(caption)
        
        # Instellingen voor de linker zijbalk en het speelveld (grid)
        self.left_sidebar_width = 300
        self.game_area_width = WIDTH - self.left_sidebar_width
        
        # Standaard gridgrootte (kan aangepast worden met toetsen 4,5,6)
        self.rows = 11
        self.cols = 11
        
        # Bereken de hexagonale celgrootte op basis van de beschikbare breedte
        self.hex_size = self.game_area_width // (self.cols + 1)
        
        # Laad de kattenafbeelding en schaal deze naar 66% van de hex celgrootte
        try:
            self.cat_image = pygame.image.load("images/cat_icon4.png").convert_alpha()
        except Exception as e:
            print("Fout bij laden van afbeelding. Zorg dat 'images/cat_icon4.png' bestaat.", e)
            raise e
        scale = int(self.hex_size * 0.66)
        self.cat_image = pygame.transform.scale(self.cat_image, (scale, scale))
        
        self.num_blocks = 10  # Standaard aantal blokken
        self.running = True
        self.ai_mode = ai_mode
        self.reset_game()

    def reset_game(self):
        # Update de hex_size op basis van de huidige gridgrootte
        self.hex_size = self.game_area_width // (self.cols + 1)
        scale = int(self.hex_size * 0.66)
        self.cat_image = pygame.transform.scale(self.cat_image, (scale, scale))
        
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.cat_pos = (self.rows // 2, self.cols // 2)
        self.game_over = False
        self.cat_trapped = False   # Houdt bij of de kat vastzit
        self.first_move = False    # Geeft aan of er al een zet is gedaan
        self.win = None            # True als gewonnen, False als verloren
        self.init_blocks()

    def init_blocks(self):
        # Plaats het vooraf ingestelde aantal blokken willekeurig
        for _ in range(self.num_blocks):
            while True:
                row = random.randint(0, self.rows - 1)
                col = random.randint(0, self.cols - 1)
                if (row, col) != self.cat_pos and self.grid[row][col] == 0:
                    self.grid[row][col] = 1
                    break

    def get_grid_offsets(self):
        """
        Berekent de offsets zodat het grid gecentreerd staat binnen het game-gebied (rechts van de zijbalk).
        """
        grid_draw_width = self.cols * self.hex_size + self.hex_size // 2
        grid_draw_height = self.rows * self.hex_size
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
        """Tekent de instructies en huidige instellingen in de linker zijbalk."""
        font = pygame.font.SysFont(None, 24)
        instructions = [
            "Controls:",
            "Click: plaats blok (human mode)",
            "R: reset spel",
            "1: 10 blokken, 2: 20 blokken, 3: 30 blokken",
            "4: grid 7x7, 5: grid 11x11, 6: grid 15x15",
            "A: toggle AI mode",
            "",
            "Goal:",
            "Trap de Kat!",
            "",
            f"Blokken: {self.num_blocks}",
            f"Grid: {self.rows}x{self.cols}",
            "",
            f"AI Mode: {'Aan' if self.ai_mode else 'Uit'}"
        ]
        x = 10
        y = 10
        for line in instructions:
            text = font.render(line, True, BLACK)
            self.screen.blit(text, (x, y))
            y += text.get_height() + 5

    def draw_grid(self):
        self.screen.fill(WHITE)
        self.draw_instructions()
        
        grid_offset_x, grid_offset_y = self.get_grid_offsets()
        # Teken het grid (speelveld) in het rechtergedeelte
        for row in range(self.rows):
            for col in range(self.cols):
                x = grid_offset_x + col * self.hex_size + (row % 2) * (self.hex_size // 2)
                y = grid_offset_y + row * self.hex_size
                color = GRAY if self.grid[row][col] == 1 else WHITE
                pygame.draw.polygon(self.screen, color, self.hexagon_points(x, y))
                pygame.draw.polygon(self.screen, BLACK, self.hexagon_points(x, y), 1)
        
        # Teken de kat met de geladen afbeelding
        cat_x = grid_offset_x + self.cat_pos[1] * self.hex_size + (self.cat_pos[0] % 2) * (self.hex_size // 2)
        cat_y = grid_offset_y + self.cat_pos[0] * self.hex_size
        image_rect = self.cat_image.get_rect(center=(cat_x + self.hex_size // 2, cat_y + self.hex_size // 2))
        self.screen.blit(self.cat_image, image_rect)

    def move_cat(self):
        """
        Beweegt de kat volgens het A*-pad. Als de kat de rand bereikt, ontsnapt hij en verlies je.
        Als er geen geldig A*-pad is, schakelt hij over naar willekeurige zetten.
        """
        if (self.cat_pos[0] == 0 or self.cat_pos[0] == self.rows - 1 or
            self.cat_pos[1] == 0 or self.cat_pos[1] == self.cols - 1):
            self.game_over = True
            self.win = False
            print("Cat escaped! Game Over!")
            return
        
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
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] == 0
    
    def is_adjacent(self, pos1, pos2):
        row, col = pos1
        directions = DIRECTIONS_EVEN if row % 2 == 0 else DIRECTIONS_ODD
        return pos2 in [(row + d[0], col + d[1]) for d in directions]
    
    def a_star(self, start):
        def heuristic(pos):
            return min(pos[0], self.rows - 1 - pos[0], pos[1], self.cols - 1 - pos[1])
        
        open_set = [(heuristic(start), 0, start, [])]
        visited = set()
        while open_set:
            _, cost, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            if (current[0] == 0 or current[0] == self.rows - 1 or
                current[1] == 0 or current[1] == self.cols - 1):
                return path + [current]
            directions = DIRECTIONS_EVEN if current[0] % 2 == 0 else DIRECTIONS_ODD
            for d in directions:
                nr, nc = current[0] + d[0], current[1] + d[1]
                if self.is_valid_move((nr, nc)) and (nr, nc) not in visited:
                    heapq.heappush(open_set, (cost + 1 + heuristic((nr, nc)),
                                               cost + 1, (nr, nc), path + [current]))
        return []
    
    def get_neighbors(self, pos, grid):
        """Geeft de aangrenzende lege cellen van pos in het gegeven grid."""
        row, col = pos
        directions = DIRECTIONS_EVEN if row % 2 == 0 else DIRECTIONS_ODD
        neighbors = []
        for d in directions:
            nr, nc = row + d[0], col + d[1]
            if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def get_reachable_area_count(self, grid, start):
        """Bereken het aantal cellen dat vanaf start bereikbaar is (via BFS)."""
        visited = set()
        queue = [start]
        visited.add(start)
        while queue:
            current = queue.pop(0)
            for neighbor in self.get_neighbors(current, grid):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited)

    def improved_ai_place_block(self):
        """
        Verbeterde AI voor het plaatsen van blokken.
        Deze versie beperkt de kandidaten tot cellen rond het huidige A*-pad van de kat
        en evalueert elke kandidaat op basis van de verhoging van de A*-padlengte en
        de vermindering van het bereikbare gebied.
        """
        # Huidige evaluaties:
        current_area = self.get_reachable_area_count(self.grid, self.cat_pos)
        current_path = self.a_star(self.cat_pos)
        current_path_length = len(current_path) if current_path else float('inf')

        # Verzamelen van kandidaten: beperk tot cellen op het pad en hun buren.
        candidate_set = set()
        if current_path:
            for pos in current_path:
                candidate_set.add(pos)
                directions = DIRECTIONS_EVEN if pos[0] % 2 == 0 else DIRECTIONS_ODD
                for d in directions:
                    neighbor = (pos[0] + d[0], pos[1] + d[1])
                    if (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols and
                        self.grid[neighbor[0]][neighbor[1]] == 0 and neighbor != self.cat_pos):
                        candidate_set.add(neighbor)
        else:
            # Geen A*-pad, overweeg dan alle lege cellen
            for row in range(self.rows):
                for col in range(self.cols):
                    if self.grid[row][col] == 0 and (row, col) != self.cat_pos:
                        candidate_set.add((row, col))

        best_score = -float('inf')
        best_move = None

        for move in candidate_set:
            # Simuleer het plaatsen van een blok op deze positie
            self.grid[move[0]][move[1]] = 1
            new_area = self.get_reachable_area_count(self.grid, self.cat_pos)
            new_path = self.a_star(self.cat_pos)
            new_path_length = len(new_path) if new_path else float('inf')
            self.grid[move[0]][move[1]] = 0

            # Als het plaatsen van het blok de kat volledig afsluit, doe dat direct
            if new_area == 1:
                print(f"Improved AI kiest directe winnende zet op {move}")
                return move

            # Bepaal een score op basis van de toename in padlengte en afname in bereikbare cellen
            score = (new_path_length - current_path_length) + (current_area - new_area)
            if score > best_score:
                best_score = score
                best_move = move

        # Fallback: als er geen kandidaat is gevonden
        if best_move is None:
            for row in range(self.rows):
                for col in range(self.cols):
                    if self.grid[row][col] == 0 and (row, col) != self.cat_pos:
                        best_move = (row, col)
                        break
                if best_move:
                    break

        print(f"Improved AI kiest zet {best_move} met score {best_score}")
        return best_move

    def handle_click(self, pos):
        grid_offset_x, grid_offset_y = self.get_grid_offsets()
        for row in range(self.rows):
            for col in range(self.cols):
                x = grid_offset_x + col * self.hex_size + (row % 2) * (self.hex_size // 2)
                y = grid_offset_y + row * self.hex_size
                hex_points = self.hexagon_points(x, y)
                if pygame.draw.polygon(self.screen, WHITE, hex_points).collidepoint(pos):
                    if (row, col) != self.cat_pos and self.grid[row][col] == 0:
                        self.grid[row][col] = 1
                        self.first_move = True
                        self.move_cat()
                    return

    def run(self):
        clock = pygame.time.Clock()
        ai_move_timer = 0  # Timer voor AI-zet als AI mode actief is
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # In human mode reageert het spel op muisklikken
                elif not self.ai_mode and event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_a:
                        self.ai_mode = not self.ai_mode
                        print("AI mode turned", "on" if self.ai_mode else "off")
                        self.reset_game()
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
                        elif event.key == pygame.K_4:
                            self.rows, self.cols = 7, 7
                            print("Grid size ingesteld op 7x7")
                            self.reset_game()
                        elif event.key == pygame.K_5:
                            self.rows, self.cols = 11, 11
                            print("Grid size ingesteld op 11x11")
                            self.reset_game()
                        elif event.key == pygame.K_6:
                            self.rows, self.cols = 15, 15
                            print("Grid size ingesteld op 15x15")
                            self.reset_game()
            
            # In AI mode laat de computer automatisch een zet doen
            if self.ai_mode and not self.game_over:
                ai_move_timer += clock.get_time()
                if ai_move_timer > 500:  # Elke 0.5 seconde
                    move = self.improved_ai_place_block()
                    if move is not None:
                        self.grid[move[0]][move[1]] = 1
                        self.first_move = True
                        self.move_cat()
                    ai_move_timer = 0

            self.draw_grid()
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
            clock.tick(30)
        pygame.quit()

if __name__ == "__main__":
    # Start het spel met AI-block placement door ai_mode=True te zetten.
    Game(ai_mode=False).run()