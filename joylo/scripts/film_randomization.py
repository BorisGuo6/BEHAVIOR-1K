import os
import time
import torch as th
import numpy as np
from typing import Dict, Optional
import json

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.envs import DataCollectionWrapper
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.robots.r1 import R1
from omnigibson.robots.r1pro import R1Pro
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.tasks import BehaviorTask
from omnigibson.systems.system_base import BaseSystem
from omnigibson.systems.macro_particle_system import MacroVisualParticleSystem
from omnigibson.utils.teleop_utils import OVXRSystem
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.object_states import Filled
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.usd_utils import GripperRigidContactAPI, ControllableObjectViewAPI
import omnigibson.utils.transform_utils as T
from omnigibson.utils.config_utils import parse_config
from omnigibson.utils.python_utils import recursively_convert_to_torch
import imageio
import time

from gello.robots.sim_robot.zmq_server import ZMQRobotServer, ZMQServerThread

from gello.robots.sim_robot.og_teleop_cfg import *
import gello.robots.sim_robot.og_teleop_utils as utils

CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160
APERTURE = 35.0
# TASK_NAME = "clearing_food_from_table_into_fridge"
# TASK_NAME = "cleaning_up_plates_and_food"
TASK_NAME = "putting_dishes_away_after_cleaning"
INSTANCE_NUMBER = 30
FRAMES_PER_INSTANCE = 30

gm.DEFAULT_VIEWER_WIDTH = CAMERA_WIDTH
gm.DEFAULT_VIEWER_HEIGHT = CAMERA_HEIGHT
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_CCD = True
gm.ENABLE_HQ_RENDERING = False

def update_to_new_instance(env, inst):
    scene_model = env.task.scene_name
    tro_filename = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=inst,
    )
    tro_file_path = f"{gm.DATASET_PATH}/scenes/{scene_model}/json/{scene_model}_task_{env.task.activity_name}_instances/{tro_filename}-tro_state.json"
    # check if tro_file_path exists, if not, then presumbaly we are done
    if not os.path.exists(tro_file_path):
        print(f"Task {env.task.activity_name} instance id: {inst} does not exist")
        print("No more task instances to load, exiting...")
        return
    with open(tro_file_path, "r") as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    env.scene.reset()
    for bddl_name, obj_state in tro_state.items():
        if "agent" in bddl_name:
            # Only set pose (we assume this is a holonomic robot, so ignore Rx / Ry and only take Rz component
            # for orientation
            robot_pos = obj_state["joint_pos"][:3] + obj_state["root_link"]["pos"]
            robot_quat = T.euler2quat(th.tensor([0, 0, obj_state["joint_pos"][5]]))
            env.task.object_scope[bddl_name].set_position_orientation(robot_pos, robot_quat)
        else:
            env.task.object_scope[bddl_name].load_state(obj_state, serialized=False)
            
    # Try to ensure that all task-relevant objects are stable
    # They should already be stable from the sampled instance, but there is some issue where loading the state
    # causes some jitter (maybe for small mass / thin objects?)
    for _ in range(25):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()
    env.scene.update_initial_file()

available_tasks = utils.load_available_tasks()
task_cfg = available_tasks[TASK_NAME][0]
cfg = utils.generate_basic_environment_config(TASK_NAME, task_cfg)
robot_config = utils.generate_robot_config(TASK_NAME, task_cfg)
cfg["robots"] = [robot_config]
env = og.Environment(configs=cfg)
# og.sim.viewer_camera.set_position_orientation(position=[3.1747, 3.9357, 1.7953], orientation=[-0.1940,  0.5288,  0.7757, -0.2846])
og.sim.viewer_camera.set_position_orientation(position=[9.2008, 1.4257, 1.7865], orientation=[0.2736, 0.5528, 0.7054, 0.3492])
og.sim.step()
breakpoint()

env.robots[0].visible = False

video_writer = imageio.get_writer(f"/home/yhang/BEHAVIOR-1K/joylo/data/filming/randomization_{TASK_NAME}_{time.time()}.mp4", fps=30)

for inst in range(INSTANCE_NUMBER):
    for frame in range(FRAMES_PER_INSTANCE):
        og.sim.step()
        for _ in range(5):
            og.sim.render()
        video_writer.append_data(og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].numpy())
    update_to_new_instance(env, inst)
    env.robots[0].visible = False
    for _ in range(10):
        og.sim.step()
    

video_writer.close()
og.shutdown()