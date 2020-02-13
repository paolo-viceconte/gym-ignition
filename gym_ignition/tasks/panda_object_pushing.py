# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import abc
import gym
import numpy as np
from typing import Tuple
from gym_ignition.base import task
from gym_ignition.utils import logger
from gym_ignition.utils.typing import Action, Observation, Reward
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from gym_ignition.base.robot import robot_abc, feature_detector, robot_joints

from gym_ignition.controllers import computed_torque_fixed_base
from gym_ignition.base.controllers import Controller, PositionControllerReferences


@feature_detector
class RobotFeatures(robot_abc.RobotABC,
                    robot_joints.RobotJoints,
                    abc.ABC):
    pass


class PandaObjectPushing(task.Task, abc.ABC):

    def __init__(self, agent_rate: float, **kwargs) -> None:
        super().__init__(agent_rate=agent_rate)

        # Store the requested robot features for this task
        self.robot_features = RobotFeatures

        self.controller = None

        self.goal_position = None

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        # joint limits
        joints_limit_min = []
        joints_limit_max = []

        # Get joint limits
        for joint_name in self.robot.joint_names():
            pos_min, pos_max = self.robot.joint_position_limits(joint_name)
            joints_limit_min.append(pos_min)
            joints_limit_max.append(pos_max)

        # Convert to numpy arrays
        joints_limit_min = np.array(joints_limit_min)
        #joints_limit_max = np.array(joints_limit_max) # WRONG
        joints_limit_max = np.array([0.04, 0.04, 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # min_joint_limits
        # [-0.001 -0.001 -2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175 -2.8973]
        # max_joint_limits
        # [0.04   0.04   2.8973 1.7628 2.8973 0.0698 2.8973 3.7525 2.8973]
        # diffs wrt original model
        # [0.04   0.04   2.8973 1.7628 2.8973 "-0.0698" 2.8973 3.7525 2.8973]

        logger.debug("Creating action space")
        action_space = gym.spaces.Box(low=joints_limit_min,
                                      high=joints_limit_max,
                                      dtype=np.float32)

        logger.debug("Creating observation space")
        observation_space = gym.spaces.Box(low=joints_limit_min,
                                           high=joints_limit_max,
                                           dtype=np.float32)

        self.controller = computed_torque_fixed_base.ComputedTorqueFixedBase(
            robot=self.robot,
            urdf=self.robot.model_file,
            controlled_joints=self.robot.joint_names(),
            kp=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            kd=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            clip_torques=False)

        ok_init = self.controller.initialize()
        assert ok_init

        # # goal position
        #
        # logger.debug("Creating goal space")
        # action_space = gym.spaces.Box(low=joints_limit_min,
        #                               high=joints_limit_max,
        #                               dtype=np.float32)
        #
        # self.goal_position =  # table size: 1.5 0.8, object initial pos: 0, 0, 1.04

        # # AFTER, extend observation for instance like:
        # high = np.array(
        #     [1.,  # cos(theta)
        #      1.,  # sin(theta)
        #      self._max_speed])
        # observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

        return action_space, observation_space

    def set_action(self, action: Action) -> bool:
        # Validate the action
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        position_reference = action

        references = PositionControllerReferences(
            position=position_reference,
            velocity=np.zeros_like(position_reference),
            acceleration=np.zeros_like(position_reference),
        )

        self.controller.set_control_references(references=references)

        torques = self.controller.step()

        for idx, joint_name in enumerate(self.robot.joint_names()):
            ok_torque = self.robot.set_joint_force(joint_name, torques[idx])
            assert ok_torque, "Failed to set the action (position reference) for the panda"

        return True

    def get_observation(self) -> Observation:

        # Get the new panda positions (maybe later velocities)
        positions_after_step = np.array(self.robot.joint_positions())

        # Create the observation object
        observation = Observation(positions_after_step)

        # Return the observation
        return observation

    def get_reward(self) -> Reward:  # TODO

        # # This environments is done only if the observation goes outside its limits.
        # # Since it can happen only when velocity is too high, penalize this happening.
        # if self.is_done():
        #     return Reward(-10000)
        #
        # # Get the data from the robot object
        # theta = self.robot.joint_position("pivot")
        # theta_dot = self.robot.joint_velocity("pivot")
        #
        # cost = \
        #     theta * theta + \
        #     0.1 * theta_dot * theta_dot + \
        #     0.001 * self._last_a
        #
        # return Reward(-cost)

        return 10

    def is_done(self) -> bool:  # TODO

        # if not self.observation_space.contains(self.get_observation()):
        #     logger.warn("Observation is outside its space. Marking the episode as done.")
        #     return True
        #
        # # This environment is episodic and always reach the max_episode_steps

        return False

    def reset_task(self) -> bool:  # TODO

        joint_positions = self.action_space.sample()
        # joint_positions = [0, 0, -2.8973, -1.7628, -2.8973, -2, 2.8973, 3.7525, -2.8973] # initialized in collision

        # Reset the robot state
        for idx, joint_name in enumerate(self.robot.joint_names()):
            ok_reset = self.robot.reset_joint(joint_name, float(joint_positions[idx]), 0.0)
            assert ok_reset, "Failed to reset the panda"

        # in order to check if the robot has been initialized in collision, you need a forward kinematics class
        # here (directly checking the contacts needs a physical step which we cannot perform here)

        # reset controller (a new robot needs a new controller)
        self.controller = computed_torque_fixed_base.ComputedTorqueFixedBase(
            robot=self.robot,
            urdf=self.robot.model_file,
            controlled_joints=self.robot.joint_names(),
            kp=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            kd=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            clip_torques=False)

        ok_init = self.controller.initialize()
        assert ok_init

        return True
