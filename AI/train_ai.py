from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from game import Game, TrapTheCatEnv

# Creëer de omgeving
game = Game()
env = TrapTheCatEnv(game)
vec_env = make_vec_env(lambda: env, n_envs=1)

# Train een DQN-agent
model = DQN("MlpPolicy", vec_env, verbose=1, learning_rate=0.0005, exploration_fraction=0.1)
model.learn(total_timesteps=20000)

# Sla het model op
model.save("trap_the_cat_dqn")
print("AI training voltooid en model opgeslagen!")
