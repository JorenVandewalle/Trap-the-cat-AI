from stable_baselines3 import DQN
from game import Game, TrapTheCatEnv
import pygame

# Laad het getrainde model
game = Game()
env = TrapTheCatEnv(game)
model = DQN.load("trap_the_cat_dqn")

# AI speelt het spel
obs, _ = env.reset()  # Reset de omgeving en pak de observation
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)  # Laat de AI een actie kiezen
    obs, reward, done, _, _ = env.step(action)  # Voer de actie uit
    env.render()  # Render het spel

    pygame.time.delay(500)  # Vertraging van 500 milliseconden (0,5 seconde)

    if done:
        print(f"Episode beëindigd. Beloning: {reward}")
        break