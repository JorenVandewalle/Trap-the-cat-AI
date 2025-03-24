import pygame
import random
import heapq
import math
import threading
import time
from collections import deque

# Constants voor het venster en kleuren
WIDTH, HEIGHT = 1200, 1000
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (200, 200, 200)
BLUE  = (50, 50, 255)
RED   = (255, 0, 0)

# Richtingen voor hexagonale beweging (afhankelijk van even/oneven rijen)
DIRECTIONS_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
DIRECTIONS_ODD  = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

class Game:
    def __init__(self, ai_mode=False):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        caption = "Trap the Cat - AI Block Placer (Minimax)" if ai_mode else "Trap the Cat"
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
        self.depth_limit = 3  # Standaard diepte voor de minimax-tree
        self.running = True
        self.ai_mode = ai_mode
        
        # Variabelen voor AI-thread en voortgang
        self.ai_thread = None
        self.ai_best_move = None
        self.ai_progress_count = 0
        self.cancel_ai = False  # Vlag om lopende AI-berekening te annuleren
        
        self.reset_game()

    def reset_game(self):
        # Annuleer eventuele lopende AI-berekening
        self.stop_ai()

        # Update de hex_size op basis van de huidige gridgrootte
        self.hex_size = self.game_area_width // (self.cols + 1)
        scale = int(self.hex_size * 0.66)
        self.cat_image = pygame.transform.scale(self.cat_image, (scale, scale))
        
        # Gridwaarden:
        # 0 = leeg, 1 = initieel geblokkeerd (grijs), 2 = door minimax gebruikte zet (blauw)
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.cat_pos = (self.rows // 2, self.cols // 2)
        self.game_over = False
        self.cat_trapped = False   # Houdt bij of de kat vastzit
        self.first_move = False    # Geeft aan of er al een zet is gedaan
        self.win = None            # True als gewonnen, False als verloren
        self.init_blocks()
        
        # Reset AI-gerelateerde variabelen
        self.ai_thread = None
        self.ai_best_move = None
        self.ai_progress_count = 0
        self.cancel_ai = False
        
        # Voor flash-effecten tijdens AI-zoektocht: set van (row, col)
        self.flash_moves = set()
        self.flash_lock = threading.Lock()

    def stop_ai(self):
        """Zet de cancel_vlag zodat de AI-thread netjes stopt en wacht even tot de thread klaar is."""
        self.cancel_ai = True
        if self.ai_thread is not None and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=0.1)
        self.ai_thread = None
        self.cancel_ai = False

    def init_blocks(self):
        # Plaats het vooraf ingestelde aantal blokken willekeurig (initieel grijs, dus waarde 1)
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
            "7/8: pas minimax diepte aan",
            "",
            "Goal:",
            "Trap de Kat!",
            "",
            f"Blokken: {self.num_blocks}",
            f"Grid: {self.rows}x{self.cols}",
            f"Minimax diepte: {self.depth_limit}",
            "",
            f"AI Mode: {'Aan' if self.ai_mode else 'Uit'}",
            f"AI Progress: {self.ai_progress_count} knopen"
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
                # Kies kleur: 
                # 1 = initieel geblokkeerd (grijs)
                # 2 = door minimax gebruikte zet (blauw)
                # Als de cel tijdelijk in flash_moves zit, teken hem rood
                with self.flash_lock:
                    if (row, col) in self.flash_moves:
                        color = RED
                    elif self.grid[row][col] == 1:
                        color = GRAY
                    elif self.grid[row][col] == 2:
                        color = BLUE
                    else:
                        color = WHITE
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
    
    # --- Minimax met progress indicator, flash-effect en cancel-check ---
    def minimax_ai_place_block(self, depth_limit=4):
        """
        Berekent de beste zet voor de AI met behulp van minimax (met alfa-beta-snoeien).
        Er wordt een voortgangscounter bijgehouden en per kandidaat zet blijft deze rood gemarkeerd
        tot de evaluatie voor die kandidaat is afgerond.
        Als self.cancel_ai True is, wordt de berekening afgebroken.
        """
        progress = {"nodes": 0}  # Voor voortgang

        def on_border(pos):
            r, c = pos
            return r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1

        def shortest_path_length(grid, cat_pos):
            queue = deque()
            visited = set()
            queue.append((cat_pos, 0))
            visited.add(cat_pos)
            while queue:
                if self.cancel_ai:
                    return math.inf
                pos, dist = queue.popleft()
                if on_border(pos):
                    return dist
                r, c = pos
                directions = DIRECTIONS_EVEN if r % 2 == 0 else DIRECTIONS_ODD
                for d in directions:
                    nr, nc = r + d[0], c + d[1]
                    new_pos = (nr, nc)
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == 0 and new_pos not in visited:
                        visited.add(new_pos)
                        queue.append((new_pos, dist + 1))
            return math.inf

        def get_valid_moves_static(grid, pos):
            r, c = pos
            moves = []
            directions = DIRECTIONS_EVEN if r % 2 == 0 else DIRECTIONS_ODD
            for d in directions:
                nr, nc = r + d[0], c + d[1]
                if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == 0:
                    moves.append((nr, nc))
            return moves

        def get_reachable_cells_static(grid, pos):
            visited = set()
            queue = deque([pos])
            visited.add(pos)
            while queue:
                if self.cancel_ai:
                    return visited
                current = queue.popleft()
                r, c = current
                directions = DIRECTIONS_EVEN if r % 2 == 0 else DIRECTIONS_ODD
                for d in directions:
                    nr, nc = r + d[0], c + d[1]
                    new_pos = (nr, nc)
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == 0 and new_pos not in visited:
                        visited.add(new_pos)
                        queue.append(new_pos)
            return visited

        def evaluate_state(grid, cat_pos):
            progress["nodes"] += 1  # update progress
            if self.cancel_ai:
                return 0
            if on_border(cat_pos):
                return -1000
            if not get_valid_moves_static(grid, cat_pos):
                return 1000
            reachable = len(get_reachable_cells_static(grid, cat_pos))
            if reachable == 1:
                return 1000
            dist = shortest_path_length(grid, cat_pos)
            if dist == math.inf:
                return 1000
            return dist

        def minimax(grid, cat_pos, depth, is_ai, alpha, beta):
            progress["nodes"] += 1  # update progress
            if self.cancel_ai:
                return 0
            if depth == 0:
                return evaluate_state(grid, cat_pos)
            if on_border(cat_pos):
                return -1000
            if not get_valid_moves_static(grid, cat_pos):
                return 1000

            current_reachable = get_reachable_cells_static(grid, cat_pos)
            restrict_moves = len(current_reachable) <= 10

            if is_ai:
                value = -math.inf
                candidate_moves = []
                if restrict_moves:
                    for pos in current_reachable:
                        r, c = pos
                        if grid[r][c] == 0 and pos != cat_pos:
                            candidate_moves.append(pos)
                else:
                    for r in range(self.rows):
                        for c in range(self.cols):
                            if grid[r][c] == 0 and (r, c) != cat_pos:
                                candidate_moves.append((r, c))
                for move in candidate_moves:
                    if self.cancel_ai:
                        return 0
                    r, c = move
                    grid[r][c] = 1  # simuleer zet
                    score = minimax(grid, cat_pos, depth - 1, False, alpha, beta)
                    grid[r][c] = 0  # herstel
                    value = max(value, score)
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
                return value
            else:
                value = math.inf
                moves = get_valid_moves_static(grid, cat_pos)
                if not moves:
                    return 1000
                for move in moves:
                    if self.cancel_ai:
                        return 0
                    score = minimax(grid, move, depth - 1, True, alpha, beta)
                    value = min(value, score)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return value

        best_move = None
        best_value = -math.inf
        candidate_moves = []
        current_reachable = get_reachable_cells_static(self.grid, self.cat_pos)
        if len(current_reachable) <= 10:
            for pos in current_reachable:
                r, c = pos
                if self.grid[r][c] == 0 and pos != self.cat_pos:
                    candidate_moves.append(pos)
        else:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r][c] == 0 and (r, c) != self.cat_pos:
                        candidate_moves.append((r, c))

        for move in candidate_moves:
            if self.cancel_ai:
                return None
            # Markeer de kandidaat cel rood en laat deze rood blijven tijdens de evaluatie
            with self.flash_lock:
                self.flash_moves.add(move)
            r, c = move
            self.grid[r][c] = 1  # simuleer zet (tijdelijk)
            value = minimax(self.grid, self.cat_pos, depth_limit - 1, False, -math.inf, math.inf)
            self.grid[r][c] = 0  # herstel
            # Verwijder de flash na de evaluatie van deze kandidaat
            with self.flash_lock:
                self.flash_moves.discard(move)
            if value > best_value:
                best_value = value
                best_move = move

        self.ai_progress_count = progress["nodes"]
        print(f"Minimax AI kiest zet {best_move} met evaluatie {best_value}")
        return best_move

    def handle_click(self, pos):
        grid_offset_x, grid_offset_y = self.get_grid_offsets()
        for row in range(self.rows):
            for col in range(self.cols):
                x = grid_offset_x + col * self.hex_size + (row % 2) * (self.hex_size // 2)
                y = grid_offset_y + row * self.hex_size
                hex_points = self.hexagon_points(x, y)
                if pygame.draw.polygon(self.screen, WHITE, hex_points).collidepoint(pos):
                    # Alleen klikken als het vakje leeg is (waarde 0) en niet de kat
                    if (row, col) != self.cat_pos and self.grid[row][col] == 0:
                        self.grid[row][col] = 1
                        self.first_move = True
                        self.move_cat()
                    return

    def start_ai_thread(self):
        # Start een nieuwe thread voor de AI-berekening
        def compute_move():
            move = self.minimax_ai_place_block(depth_limit=self.depth_limit)
            self.ai_best_move = move
        self.ai_thread = threading.Thread(target=compute_move)
        self.ai_thread.start()

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif not self.ai_mode and event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    # Bij elke wijziging zorgen we dat de AI wordt geannuleerd
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_a:
                        self.ai_mode = not self.ai_mode
                        print("AI mode turned", "on" if self.ai_mode else "off")
                        self.reset_game()
                    elif event.key == pygame.K_7:
                        self.depth_limit = max(1, self.depth_limit - 1)
                        print("Minimax diepte verlaagd naar", self.depth_limit)
                        self.reset_game()
                    elif event.key == pygame.K_8:
                        self.depth_limit += 1
                        print("Minimax diepte verhoogd naar", self.depth_limit)
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
            
            # In AI mode starten we periodiek een thread indien nodig
            if self.ai_mode and not self.game_over:
                if self.ai_thread is None or not self.ai_thread.is_alive():
                    if self.ai_best_move is not None:
                        # Als de thread klaar is, pas de zet toe en markeer de zet als blauw (waarde 2)
                        move = self.ai_best_move
                        if move is not None:
                            r, c = move
                            self.grid[r][c] = 2  # markeer zet als door AI gebruikt (blauw)
                            self.first_move = True
                            self.move_cat()
                        self.ai_best_move = None
                    else:
                        # Start een nieuwe AI-thread als er geen actief is
                        self.start_ai_thread()

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
    # Start het spel met AI-block placement in minimax modus
    Game(ai_mode=False).run()
