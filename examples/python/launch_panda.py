#!/usr/bin/env python3

# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import gym
import numpy as np

# Set gym verbosity
gym.logger.set_level(gym.logger.INFO)
assert gym.logger.set_level(gym.logger.DEBUG) or True

# Register gym-ignition environments
import gym_ignition
from gym_ignition.utils import logger

# Create the environment
env = gym.make("PandaObjectPushing-Gazebo-v0")

# Enable the rendering
env.render('human')

# # Initialize the seed
# env.seed(42)

for epoch in range(10):

    #print("NEW_EPOCH")

    # Reset the environment
    observation = env.reset()

    #print("reset done")
    #print(env.action_space)
    #print(env.observation_space)

    # Initialize returned values
    done = False
    totalReward = 0
    step_counter=0
    rollout_termination_counter=0

    while not done:
        # Execute a random action
        action = env.action_space.sample()

        #print(step_counter)
        step_counter+=1

        observation, reward, done, _ = env.step(action)  # it gets stuck here sometimes

        # Render the environment
        # It is not required to call this in the loop
        # env.render('human')

        # Accumulate the reward
        totalReward += reward

        # Print the observation
        msg = ""
        for value in observation:
            msg += "\t%.6f" % value
        # logger.debug(msg)

        rollout_termination_counter+=1
        if (rollout_termination_counter>1000):
            done=True

#     logger.info(f"Total reward for episode #{epoch}: {totalReward}")

env.close()
