from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from game import Game, TrapTheCatEnv

game = Game()
env = TrapTheCatEnv(game)
vec_env = make_vec_env(lambda: env, n_envs=1)

model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    tensorboard_log="./logs"
)

model.learn(total_timesteps=100000, progress_bar=True)
model.save("trap_the_cat_ppo")
print("Training succesvol voltooid!")