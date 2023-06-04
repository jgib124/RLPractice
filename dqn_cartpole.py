"""
Practice of Deep Q-Learning with cartpole game
Using conda environment

5/25/23
"""

import numpy as np
import tensorflow as tf
import gymnasium as gym

# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adams

# from rl.agents import DQNAgent # Deep Q-Learning Agent
# from rl.policy import BoltzmannQPolicy # Policy
# from rl.memory import SequentialMemory # Memory

# Use stablebaseline3 instead of keras-rl
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import keras 
import random
import time
import os

# First pass through DNN is to get the Q values for each action - POLICY NETWORK
# The second pass gives us Q-values that can be reached from the new (s, a) pair - TARGET NETWORK
# The target network is updated every C steps, and is used to calculate loss for the policy network


# --- SET UP ENVIRONMENT --- 
env = gym.make('CartPole-v1', render_mode="human")
states = env.observation_space.shape[0]
actions = env.action_space.n

print("States: ", states, type(env.observation_space), "\nActions: ", actions, type(env.action_space))
# print(spaces.Box(env.action_space))
time.sleep(1)



# --- RANDOM AGENT ---
# choosing random actions in environment as a baseline comparison
print("\n---RANDOM AGENT---\n")
episodes = 10
max_num_steps = 250
random_scores = []
for episode in range(1, episodes + 1):
    state = env.reset()
    terminated, truncated = False, False
    score = 0

    while not terminated and not truncated:
        env.render()
        # choose a randomy action within the 
        action = random.choice([0, 1])

        # Render state of environment to display
        env.render() # Renders with pygame

        new_state, reward, terminated, truncated, info = env.step(action)
        # print("State: ", new_state, "Action: ", action, "Reward: ", reward)

        score += reward

    print(f"Episode: {episode} Score: {score}")
    random_scores.append(score)




# --- DEEP Q-LEARNING AGENT ---

# Define function that can build a deep neural network
# Num states == num input nodes
# Num actions == num output nodes
# def build_dnn_model(num_states, num_actions):
#     model = Sequential()
#     model.add(Flatten(input_shape=(1, num_states)))
#     model.add(Dense(24, activation='relu'))
#     model.add(Dense(24, activation='relu'))
#     model.add(Dense(num_actions, activation='linear'))

#     return model


# def build_rl_agent(mode, actions):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=50000, window_length=1)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy,
#                     nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)

#     return dqn


# # build a policy model for the cartpole environment
# model = build_dnn_model(states, actions)
# print(model.summary())
# time.sleep(1)

# # build a deep q-learning agent using the policy model and the observation space
# dqn = build_rl_agent(model, actions)
# Adam._name = "Jimbo & Smellie's Optimizer!!!"
# dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=50000, visualize=True, verbose=1)

# --- STABLE BASELINES3 AGENT ---

check_env(env, warn=True)

print("\n---STABLE BASELINES3 AGENT---\n")

# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=50_000)
# model.save("ppo_cartpole")

model = PPO.load("ppo_cartpole", env=env)

vec_env = model.get_env()

number_episodes = 10
dqn_scores = []

for episode in range(1, number_episodes+1):
    observation = vec_env.reset()
    score = 0
    done = False

    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = vec_env.step(action)
        vec_env.render()

        score += reward

    print(f"Episode: {episode} Score: {score}")



