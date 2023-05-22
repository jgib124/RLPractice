"""
Practice of Q-learning with FrozenLake-v1
Using conda environment

James Gibbons
5/22/23
"""

import numpy as np
import gym
import random
import time

# Create the environment from the gym library
env = gym.make("FrozenLake-v1")

# Create the Q-table
# We can query the action space and state space
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

# Instantiate the Q-table to the size of all (s, a) pairs
q_table = np.zeros((state_space_size, action_space_size))

# Print the Q-table to terminal
print(q_table)

# ---HYPERPARAMETERS---
# Number of episodes = number of times we run the environment from the beginning
num_episodes = 10000
# Max number of steps per episode = max number of steps that the agent will take within a single episode
max_steps_per_episode = 100

# ---LEARNING PARAMETERS---
# Learning rate = weight of updated reward information vs. previous Q-value
learning_rate = 0.1
# Discount rate = discount factor of future rewards
discount_rate = 0.99

# ---EXPLORATION PARAMETERS---
# Exploration rate = rate at which the agent randomly decides its action rather than using the policy
exploration_rate = 1
# Max exploration rate = max rate at which the agent can explore
max_exploration_rate = 1
# Min exploration rate = min rate at which the agent can explore
min_exploration_rate = 0.01
# Exploration decay rate = rate at which the agent will decay the exploration rate towards the min exploration rate
exploration_decay_rate = 0.001


# ---Q-LEARNING ALGORITHM---
# create list to track rewards from each episode
rewards_all_episodes = []

# Q-learning algorithm
# Each loop iteration runs a new episode
for episode in range(num_episodes):
    state, _ = env.reset()  # reset the environment to start a new episode

    terminated = False  # boolean to track if the episode is finished
    truncated = False  # boolean to track if the episode is finished
    rewards_current_episode = 0  # variable to track the total reward within the current episode

    # Each loop iteration runs a new step within the episode
    for step in range(max_steps_per_episode):

        # Exploration-exploitation trade-off
        # Generate a random number between 0 and 1 to decide between exploration and exploitation
        exploration_rate_threshold = random.uniform(0, 1)
        # If the random number is greater than the exploration rate, we will exploit
        if exploration_rate_threshold > exploration_rate:
            # Can only choose actions available from the current state
            action = np.argmax(q_table[state, :])
        else:
            # If the random number is less than the exploration rate, we will explore
            action = env.action_space.sample()

        # Using the action, generate the next state, reward, whether the episode is done,
        # and additional info from the environment
        # NOTE: terminated is if the episode is done
        # NOTE: truncated is if the episode is stopped from a non-MDP reason
        new_state, reward, terminated, truncated , info = env.step(action)

        # Update Q-table for Q(s, a)
        # NOTE: max[Q(s', a')] is the maximum possible Q-value for all actions from the new state
        # Q_new(s, a) = [(1 - learning_rate)*Q(s, a)]
        #  + [learning_rate * (reward + discount_rate * max[Q(s', a')]]
        q_table[state, action] = ((1 - learning_rate) * q_table[state, action]) + \
            (learning_rate * (reward + (discount_rate * np.max(q_table[new_state, :]))))
        
        # Update to new state and update total reward
        state = new_state
        rewards_current_episode += reward

        # Check if the episode is done
        if terminated == True or truncated == True:
            break

    # Loop: STEPS

    # After each episode, decay the exploration rate
    # Increases exploitation of q-table
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)


    # Append the current episode reward to the list of all episode rewards
    rewards_all_episodes.append(rewards_current_episode)

# Loop: EPISODES
# Calculate and print the average reward per thousand episodes
# Split array into chunks of 1000
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
print("Average reward per thousand episodes:\n")
count = 1000
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\nQ-table:\n")
print(q_table)





 



