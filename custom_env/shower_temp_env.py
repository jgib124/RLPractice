# Create custom environment for shower temperature control
# Optimal Range = 37 - 39
# Actions: turn up, turn down, do nothing

# Need to define 4 functions: init, step, render, reset

import gym
import random
import numpy as np
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box

# Inherit from gym Env baseclass
class ShowerTempEnv(Env):
    # Initialization Function
    def __init__(self):
        # Three different types of actions: turn up, turn down, do nothing
        self.action_space = Discrete(3) 
        # Temperature range from 0 to 100
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.int64)
        # stochastic starting state
        self.state = 38 + random.randint(-3, 3)
        # Shower length in seconds 
        self.shower_length = 60 

    # Step Function
    def step(self, action):
        # Runs every time a step is taken in the environment
        # Show temperature decreases over time if not adjusted
        self.state += action - 1 
        # Decrement shower length
        self.shower_length -= 1

        # Return reward for being in the correct range
        reward = -1
        if self.state >=37 and self.state <=39:
            reward = 1

        # Check if shower episode is over
        terminated = False
        truncated = False
        if self.shower_length <=0:
            truncated = True

        # There is no terminal goal state in this problem

        # Add stochasticity to state
        self.state += random.randint(-1, 1)

        info = {} # requirement for openai gym

        # Gym step requires 4 values to be returned
        # New State, Reward for previous action, Done, Info
        return np.array([self.state]), reward, terminated, truncated, info


    # Render Function
    def render(self):
        # Render the environment to the screen
        print(f"Shower Temperature ({self.shower_length}): {self.state} C {' + ' if (self.state >=37 and self.state <=39) else ''}")
        pass

    # Reset Function
    def reset(self, seed=0):
        super().reset(seed=seed)
        # stochastic starting state
        self.state = 38 + random.randint(-3, 3)
        # Shower length in seconds 
        self.shower_length = 60 

        info = {} # requirement for openai gym

        return np.array([self.state]), info

