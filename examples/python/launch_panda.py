#!/usr/bin/env python3

# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import gym
import time

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
#
# for epoch in range(10):
#     # Reset the environment
#     observation = env.reset()
#
#     # Initialize returned values
#     done = False
#     totalReward = 0
#
#     while not done:
#         # Execute a random action
#         action = env.action_space.sample()
#         observation, reward, done, _ = env.step(action)
#
#         # Render the environment
#         # It is not required to call this in the loop
#         # env.render('human')
#
#         # Accumulate the reward
#         totalReward += reward
#
#         # Print the observation
#         msg = ""
#         for value in observation:
#             msg += "\t%.6f" % value
#         logger.debug(msg)
#
#     logger.info(f"Total reward for episode #{epoch}: {totalReward}")


for epoch in range(10):
    print(epoch)
    # Reset the environment
    observation = env.reset()

    # Initialize returned values
    done = False
    totalReward = 0

    while not done:
        # Execute a random action
        action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)

        # Render the environment
        # It is not required to call this in the loop
        env.render('human')

        # Accumulate the reward
        totalReward += reward

        # print("---------------ACTION-------------------")
        # print(action)
        # print("---------------OBSERVATION-------------------")
        # print(observation)
        # print("---------------REWARD-------------------")
        # print(reward)
        # print("---------------DONE-------------------")
        # print(done)
        # print("---------------TOTAL REWARD-------------------")
        # print(totalReward)
        # time.sleep(1)

        # Print the observation
        msg = ""
        for value in observation:
            msg += "\t%.6f" % value
        logger.debug(msg)

env.close()
