# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import gym
import time
import math
import numpy as np
import gympp_bindings as bindings
from gym_ignition.utils import logger
from gym_ignition.utils import resource_finder
from gym_ignition.robots.sim.gazebo import pendulum
from gym_ignition.controllers import computed_torque_fixed_base_cpp
from gym_ignition.base.controllers import Controller, PositionControllerReferences


def to_quaternion(yaw, pitch, roll):  # yaw(Z), pitch(Y), roll(X) in radians

    # abbreviations for the various angular functions
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # fill the quaternion
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    q = [w, x, y, z]

    return q


rtf = 0.1
iterations = 1
physics_rate = 500.0

gazebo = bindings.GazeboWrapper(iterations, rtf, physics_rate)
assert gazebo

gazebo.setVerbosity(4)
gym.logger.set_level(gym.logger.DEBUG)

empty_world = resource_finder.find_resource("DefaultEmptyWorld.world")
ok_world = gazebo.setupGazeboWorld(worldFile=empty_world)
assert ok_world

ok_initialize = gazebo.initialize()
assert ok_initialize

q_1 = to_quaternion(0,0,0)
robot_1 = pendulum.PendulumGazeboRobot(model_file="Pendulum/Pendulum.urdf",
                                      gazebo=gazebo,
                                      base_position=np.array([0, -1, 0]),
                                      base_orientation=np.array(q_1))

q_2 = to_quaternion(0,0,0)
robot_2 = pendulum.PendulumGazeboRobot(model_file="Pendulum/Pendulum.urdf",
                                      gazebo=gazebo,
                                      base_position=np.array([0, 1, 0]),
                                      base_orientation=np.array(q_2))

ok_gui = gazebo.gui()

controlled_joints_1 = [j for j in robot_1.joint_names() if not "_ft_" in j]

controlled_joints_2 = [j for j in robot_2.joint_names() if not "_ft_" in j]

controller_1 = computed_torque_fixed_base_cpp.ComputedTorqueFixedBaseCpp(
    robot=robot_1,
    urdf="Pendulum/Pendulum.urdf",
    controlled_joints=controlled_joints_1,
    kp=np.array([800.0]),
    kd=np.array([800.0]),
    clip_torques=False)

ok_init_1 = controller_1.initialize()
assert ok_init_1

controller_2 = computed_torque_fixed_base_cpp.ComputedTorqueFixedBaseCpp(
    robot=robot_2,
    urdf="Pendulum/Pendulum.urdf",
    controlled_joints=controlled_joints_2,
    kp=np.array([800.0]),
    kd=np.array([800.0]),
    clip_torques=False)

ok_init_2 = controller_2.initialize()
assert ok_init_2

dt = 0.01
simulation_duration = 50
num_steps = int(simulation_duration / dt)

position_reference_1 = np.array([-0.6])
position_reference_2 = np.array([0.6])

for ts in range(num_steps):
    print(f"{ts}")

    references_1 = PositionControllerReferences(
        position=position_reference_1,
        velocity=np.zeros_like(position_reference_1),
        acceleration=np.zeros_like(position_reference_1),
    )

    controller_1.set_control_references(references=references_1)

    torques_1 = controller_1.step()

    for idx, joint_name in enumerate(robot_1.joint_names()):
        ok_torque = robot_1.set_joint_force(joint_name, torques_1[idx])
        assert ok_torque

    references_2 = PositionControllerReferences(
        position=position_reference_2,
        velocity=np.zeros_like(position_reference_2),
        acceleration=np.zeros_like(position_reference_2),
    )

    controller_2.set_control_references(references=references_2)

    torques_2 = controller_2.step()

    for idx, joint_name in enumerate(robot_2.joint_names()):
        ok_torque = robot_2.set_joint_force(joint_name, torques_2[idx])
        assert ok_torque

    gazebo.run()

    positions_after_step_1 = robot_1.joint_positions()
    positions_after_step_2 = robot_2.joint_positions()

    if np.allclose(position_reference_1, positions_after_step_1, atol=0.1) and np.allclose(position_reference_2, positions_after_step_2, atol=1):
        position_reference_1 *= -1
        position_reference_2 *= -1

time.sleep(5)
gazebo.close()