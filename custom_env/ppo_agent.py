import gymnasium as gym
import numpy as np
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from shower_temp_env import ShowerTempEnv



env = ShowerTempEnv()
actions = env.action_space
observation_space = env.observation_space

print("Actions:", [env.action_space.sample() for i in range(25)])
print("Observations:", observation_space.shape)
print("Starting State:", env.state)

check_env(env, warn=True)

# Train Model
model_path = os.path.join(os.getcwd(), "custom_env", "ppo_shower_temp.zip")

print("Model Path:", model_path)
time.sleep(3)

# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=200_000)
# model.save(model_path)

model = PPO.load(model_path)

number_episodes = 10
scores = []

vec_env = model.get_env()
print("Vectorized Environment:", vec_env)

for episode in range(1, number_episodes+1):
    observation, info = env.reset()
    score = 0
    terminated, truncated = False, False

    print(f"Episode {episode} Starting State: {observation}")

    while not (terminated or truncated):
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        score += reward

    scores.append(score)
    print(f"Episode: {episode} Score: {score}")

print(f"Mean Score: {np.mean(scores)}")
