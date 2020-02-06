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

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # Store the requested robot features for this task
        self.robot_features = RobotFeatures

        # should it be here????
        self.controller = None

        # Private attributes # NEEDED????
        self._last_a = None

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
        joints_limit_max = np.array(joints_limit_max)

        print(joints_limit_min)
        print(joints_limit_max)

        logger.debug("Creating action space")
        action_space = gym.spaces.Box(low=joints_limit_min,
                                      high=joints_limit_max,
                                      dtype=np.float32)

        logger.debug("Creating observation space")
        observation_space = gym.spaces.Box(low=joints_limit_min,
                                           high=joints_limit_max,  # shape=(1,) needed?
                                           dtype=np.float32)

        # controller
        controlled_joints = [j for j in self.robot.joint_names() if not "_ft_" in j]

        self.controller = computed_torque_fixed_base.ComputedTorqueFixedBase(
            robot=self.robot,
            urdf=self.robot.model_file,
            controlled_joints=controlled_joints,
            kp=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            kd=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            clip_torques=False)

        ok_init = self.controller.initialize()
        assert ok_init

        # # AFTER, extend observation!!!!!!!!!!!!!!
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

        # Store the last action. It is used to calculate the reward.
        self._last_a = action

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

        # Get the robot object # maybe better
        #robot = self.robot

        # Get the new panda positions (maybe later velocities)
        positions_after_step = self.robot.joint_positions()

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

        # # Sample the angular velocity from the observation space
        # _, _, theta_dot = self.observation_space.sample().tolist()
        #
        # # Sample the angular position from an uniform rng
        # theta = self.np_random.uniform(0, 2 * np.pi)
        #
        # try:
        #     desired_control_mode = robot_joints.JointControlMode.TORQUE
        #     if self.robot.joint_control_mode("pivot") != desired_control_mode:
        #         ok_mode = self.robot.set_joint_control_mode("pivot", desired_control_mode)
        #         assert ok_mode, "Failed to set pendulum control mode"
        # except Exception:
        #     logger.warn("Failed to set control mode. Is it supported by the runtime?")
        #     pass
        #
        # # Reset the robot state
        # ok_reset = self.robot.reset_joint("pivot", theta, theta_dot)
        # assert ok_reset, "Failed to reset the pendulum"
        #
        # # Clean the last applied force
        # self._last_a = None

        return True
