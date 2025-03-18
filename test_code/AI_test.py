import pygame
import random
import heapq
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import os

# Constants voor het venster en kleuren
WIDTH, HEIGHT = 1200, 1000
WHITE, BLACK, GRAY, BLUE = (255, 255, 255), (0, 0, 0), (200, 200, 200), (50, 50, 255)

# Richtingen voor hexagonale beweging (afhankelijk van even/oneven rijen)
DIRECTIONS_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
DIRECTIONS_ODD  = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Trap the Cat")
        
        # Instellingen voor de linker zijbalk en het speelveld (grid)
        self.left_sidebar_width = 300
        self.game_area_width = WIDTH - self.left_sidebar_width
        
        # Standaard gridgrootte (aanpasbaar met toetsen 4,5,6)
        self.rows = 11
        self.cols = 11
        
        # Bereken de hexagonale celgrootte op basis van de beschikbare breedte
        self.hex_size = self.game_area_width // (self.cols + 1)
        
        # Probeer de kattenafbeelding te laden, zo niet, gebruik een fallback (blauwe cirkel)
        cat_img_path = os.path.join("images", "cat_icon4.png")
        if os.path.exists(cat_img_path):
            self.cat_image = pygame.image.load(cat_img_path).convert_alpha()
        else:
            print("Image 'cat_icon4.png' niet gevonden, gebruik fallback.")
            self.cat_image = None
        
        if self.cat_image:
            scale = int(self.hex_size * 0.66)
            self.cat_image = pygame.transform.scale(self.cat_image, (scale, scale))
        
        self.num_blocks = 10  # Standaard aantal blokken
        self.running = True
        self.reset_game()

    def reset_game(self):
        self.hex_size = self.game_area_width // (self.cols + 1)
        if self.cat_image:
            scale = int(self.hex_size * 0.66)
            self.cat_image = pygame.transform.scale(self.cat_image, (scale, scale))
        
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.cat_pos = (self.rows // 2, self.cols // 2)
        self.game_over = False
        self.cat_trapped = False  # Houdt bij of de kat vastzit
        self.first_move = False   # Geeft aan of er al een zet is gedaan
        self.win = None           # True als gewonnen, False als verloren
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
        font = pygame.font.SysFont(None, 24)
        instructions = [
            "Controls (Keypad):",
            "1: 10 blokken",
            "2: 20 blokken",
            "3: 30 blokken",
            "4: grid 7x7",
            "5: grid 11x11",
            "6: grid 15x15",
            "R: reset spel",
            "A: start AI solving",
            "",
            "Goal:",
            "Trap de Kat!"
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
        for row in range(self.rows):
            for col in range(self.cols):
                x = grid_offset_x + col * self.hex_size + (row % 2) * (self.hex_size // 2)
                y = grid_offset_y + row * self.hex_size
                color = GRAY if self.grid[row][col] == 1 else WHITE
                pygame.draw.polygon(self.screen, color, self.hexagon_points(x, y))
                pygame.draw.polygon(self.screen, BLACK, self.hexagon_points(x, y), 1)
        
        # Teken de kat: als afbeelding beschikbaar, anders fallback naar cirkel.
        cat_x = grid_offset_x + self.cat_pos[1] * self.hex_size + (self.cat_pos[0] % 2) * (self.hex_size // 2)
        cat_y = grid_offset_y + self.cat_pos[0] * self.hex_size
        if self.cat_image:
            image_rect = self.cat_image.get_rect(center=(cat_x + self.hex_size // 2, cat_y + self.hex_size // 2))
            self.screen.blit(self.cat_image, image_rect)
        else:
            pygame.draw.circle(self.screen, BLUE, (cat_x + self.hex_size // 2, cat_y + self.hex_size // 2), self.hex_size // 3)

    def move_cat(self):
        # Als de kat aan de rand staat, ontsnapt hij => verlies
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

    # --- Methoden voor RL integratie ---
    def get_state(self):
        state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 1:
                    state[0, r, c] = 1.0
        r, c = self.cat_pos
        state[1, r, c] = 1.0
        return state

    def reset_env(self):
        self.reset_game()
        return self.get_state()

    def step(self, action):
        if self.game_over:
            return self.get_state(), 0, True, {}

        row, col = action // self.cols, action % self.cols
        reward = -1  # standaard stapkost

        if (row, col) == self.cat_pos or self.grid[row][col] == 1:
            reward = -5
        else:
            self.grid[row][col] = 1
            self.first_move = True

        if not self.game_over:
            self.move_cat()

        done = self.game_over
        if done:
            reward = 10 if self.win else -10
        next_state = self.get_state()
        return next_state, reward, done, {}

# --- DQN Agent en ondersteunende klassen ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_shape, action_size, lr=1e-3, gamma=0.99, device='cpu'):
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.input_dim = np.prod(state_shape)
        self.policy_net = DQN(self.input_dim, action_size).to(device)
        self.target_net = DQN(self.input_dim, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.update_target_steps = 1000
        self.steps_done = 0

    def select_action(self, state_flat, valid_actions_mask):
        self.steps_done += 1
        if random.random() < self.epsilon:
            valid_indices = np.where(valid_actions_mask == 1)[0]
            if len(valid_indices) == 0:
                return random.randrange(self.action_size)
            return int(random.choice(valid_indices))
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().data.numpy().flatten()
                q_values[valid_actions_mask == 0] = -float('inf')
                return int(np.argmax(q_values))

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device).view(self.batch_size, -1)
        next_states = torch.FloatTensor(next_states).to(self.device).view(self.batch_size, -1)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(state_action_values, expected_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()

def get_valid_actions_mask(state):
    grid = state[0]
    cat_channel = state[1]
    valid_mask = ((grid == 0) & (cat_channel == 0)).astype(np.int32)
    return valid_mask.flatten()

def train_rl_agent(num_episodes=1000):
    env = Game()  # Gebruik de Game als omgeving
    state_shape = (2, env.rows, env.cols)
    action_size = env.rows * env.cols
    agent = DQNAgent(state_shape, action_size, device='cpu')
    total_steps = 0

    for episode in range(num_episodes):
        state = env.reset_env()
        done = False
        episode_reward = 0

        while not done:
            state_flat = state.flatten()
            valid_mask = get_valid_actions_mask(state)
            action = agent.select_action(state_flat, valid_mask)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            loss = agent.optimize()  # Trainingsstap
            total_steps += 1
            if total_steps % agent.update_target_steps == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        print(f"Episode {episode} - Reward: {episode_reward} - Epsilon: {agent.epsilon:.3f}")
    
    # Sla de getrainde gewichten op
    torch.save(agent.policy_net.state_dict(), "dqn_weights.pth")
    print("Training voltooid en gewichten opgeslagen als 'dqn_weights.pth'.")

def play_game_with_ai():
    """
    Start een spel waarin de AI (DQN agent) automatisch blokken plaatst.
    De AI-oplossing start pas wanneer de gebruiker op de A-toets drukt.
    Andere toetsen (R, 1,2,3,4,5,6) worden gebruikt om instellingen aan te passen en te resetten.
    """
    env = Game()
    state_shape = (2, env.rows, env.cols)
    action_size = env.rows * env.cols
    agent = DQNAgent(state_shape, action_size, device='cpu')
    
    # Probeer getrainde gewichten te laden, indien beschikbaar
    if os.path.exists("dqn_weights.pth"):
        agent.policy_net.load_state_dict(torch.load("dqn_weights.pth", map_location='cpu'))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("Getrainde gewichten geladen.")
    else:
        print("Geen getrainde gewichten gevonden. De AI gebruikt random gewichten.")
    
    # Voor de demo: standaard epsilon op 0 voor exploitatief gedrag (maar het beleid is alleen zinvol na training)
    agent.epsilon = 0.0

    state = env.reset_env()
    clock = pygame.time.Clock()
    running = True
    solving_started = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    solving_started = True
                elif event.key == pygame.K_r:
                    state = env.reset_env()
                    solving_started = False
                elif event.key == pygame.K_1:
                    env.num_blocks = 10
                    state = env.reset_env()
                    solving_started = False
                elif event.key == pygame.K_2:
                    env.num_blocks = 20
                    state = env.reset_env()
                    solving_started = False
                elif event.key == pygame.K_3:
                    env.num_blocks = 30
                    state = env.reset_env()
                    solving_started = False
                elif event.key == pygame.K_4:
                    env.rows, env.cols = 7, 7
                    state = env.reset_env()
                    solving_started = False
                elif event.key == pygame.K_5:
                    env.rows, env.cols = 11, 11
                    state = env.reset_env()
                    solving_started = False
                elif event.key == pygame.K_6:
                    env.rows, env.cols = 15, 15
                    state = env.reset_env()
                    solving_started = False
        
        if solving_started and not env.game_over:
            state_flat = state.flatten()
            valid_mask = get_valid_actions_mask(state)
            action = agent.select_action(state_flat, valid_mask)
            state, reward, done, _ = env.step(action)
        
        env.draw_grid()
        if not solving_started:
            font = pygame.font.SysFont(None, 55)
            message = 'Press A to start solving'
            text = font.render(message, True, BLACK)
            env.screen.blit(text, ((WIDTH - text.get_width()) // 2,
                                   (HEIGHT - text.get_height()) // 2))
        if env.game_over:
            font = pygame.font.SysFont(None, 55)
            if env.win:
                message = 'AI won! Cat trapped. Press R or change settings.'
            else:
                message = 'AI lost! Cat escaped. Press R or change settings.'
            text = font.render(message, True, BLACK)
            env.screen.blit(text, ((WIDTH - text.get_width()) // 2,
                                   (HEIGHT - text.get_height()) // 2))
        pygame.display.flip()
        clock.tick(5)
    pygame.quit()

if __name__ == "__main__":
    # Gebruik command-line argumenten:
    # "rl" => Training modus (zonder pygame display)
    # "ai" => AI play mode (met pygame display en toetsenbordinput)
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "rl":
            train_rl_agent(num_episodes=10000)  # Pas het aantal episodes aan naar wens
        elif mode == "ai":
            play_game_with_ai()
        else:
            print("Onbekend argument. Gebruik 'rl' voor training of 'ai' voor AI play mode.")
    else:
        play_game_with_ai()
