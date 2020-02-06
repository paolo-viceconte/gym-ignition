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
from gym_ignition.robots.sim.gazebo import panda
from gym_ignition.controllers import computed_torque_fixed_base
from gym_ignition.controllers import computed_torque_fixed_base_cpp
from gym_ignition.base.controllers import Controller, PositionControllerReferences

# for the controller only (it needs a separate argument for the time being)
import gym_ignition_models as m
panda_model = m.get_model_file('panda')

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


rtf = 1
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

print("PRE ROBOT")

q_1 = to_quaternion(0,0,0)
panda_robot = panda.PandaGazeboRobot(gazebo=gazebo,
                                     base_position=np.array([0, 1, 0]),
                                     base_orientation=np.array(q_1))

print("POST ROBOT")

ok_gui = gazebo.gui()

controlled_joints = [j for j in panda_robot.joint_names() if not "_ft_" in j]
# ['panda_finger_joint1', 'panda_finger_joint2', 'panda_joint1', 'panda_joint2',
# 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

controller = computed_torque_fixed_base.ComputedTorqueFixedBase(
    robot=panda_robot,
    urdf=panda_model,
    controlled_joints=controlled_joints,
    kp=np.array([10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0]),
    kd=np.array([10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0]),
    clip_torques=False)

ok_init = controller.initialize()
assert ok_init

dt = 0.01
simulation_duration = 500
num_steps = int(simulation_duration / dt)

position_reference = np.array([0, 0, 0, 0, 0, 0, 2, 0, 0])

# time to allow to take the video
time.sleep(5)
print("10 s to start")
time.sleep(5)
print("5 s to start")
time.sleep(5)

wait_there=0

for ts in range(num_steps):
    print(f"{ts}")

    references = PositionControllerReferences(
        position=position_reference,
        velocity=np.zeros_like(position_reference),
        acceleration=np.zeros_like(position_reference),
    )

    controller.set_control_references(references=references)

    torques = controller.step()

    for idx, joint_name in enumerate(panda_robot.joint_names()):
        ok_torque = panda_robot.set_joint_force(joint_name, torques[idx])
        assert ok_torque

    gazebo.run()

    positions_after_step = panda_robot.joint_positions()

    if np.allclose(position_reference, positions_after_step, atol=0.1):
        if wait_there<300:
            wait_there+=1
        else:
            wait_there=0
            position_reference *= -1

    print(wait_there)

time.sleep(50)
gazebo.close()
