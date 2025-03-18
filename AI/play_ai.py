from stable_baselines3 import PPO
from game import Game, TrapTheCatEnv, ROWS, COLS
import pygame
import numpy as np

game = Game()
env = TrapTheCatEnv(game)
model = PPO.load("trap_the_cat_ppo")

obs, _ = env.reset()
done = False

while not done:
    possible_actions = game.get_possible_actions()
    possible_actions_flat = [row * COLS + col for (row, col) in possible_actions]
    
    action, _ = model.predict(obs, deterministic=True)
    row, col = divmod(action, COLS)
    
    print(f"\nBeschikbare acties: {len(possible_actions)}")
    print(f"AI kiest: ({row}, {col}) - {'Geldig' if action in possible_actions_flat else 'Ongeldig'}")

    obs, reward, done, _, _ = env.step(action)
    env.render()
    
    pygame.time.delay(300)
    
    if done:
        print("\nResultaat:", "Kat gevangen!" if reward > 0 else "Kat ontsnapt!")