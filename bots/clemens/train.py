from stable_baselines3 import PPO
from env.env import Env

env = Env()
model = PPO(policy="MultiInputPolicy", env=env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("model")
