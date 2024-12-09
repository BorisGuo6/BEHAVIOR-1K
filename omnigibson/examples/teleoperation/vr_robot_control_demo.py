"""
Example script for interacting with OmniGibson scenes with VR and BehaviorRobot.
"""

import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.teleop_utils import OVXRSystem

gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = True
gm.GUI_VIEWPORT_ONLY = True

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


def main():
    """
    Spawn a BehaviorRobot in Rs_int and users can navigate around and interact with the scene using VR.
    """
    # Create the config for generating the environment we want
    scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}
    robot0_cfg = {
        "type": "R1",
        "obs_modalities": ["rgb"],
        "controller_config": {
            "arm_left": {
                "name": "InverseKinematicsController",
                "mode": "absolute_pose",
                "pos_kp": 5.0,
                "vel_kp": 0.0,
                "command_input_limits": None,
                "command_output_limits": None,
            },
            "arm_right": {
                "name": "InverseKinematicsController",
                "mode": "absolute_pose",
                "pos_kp": 5.0,
                "vel_kp": 0.0,
                "command_input_limits": None,
                "command_output_limits": None,
            },
            "gripper_left": {"name": "MultiFingerGripperController", "command_input_limits": "default"},
            "gripper_right": {"name": "MultiFingerGripperController", "command_input_limits": "default"},
        },
        "action_normalize": False,
        "reset_joint_pos": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -1.8000,
            -0.8000,
            0.0000,
            -0.0068,
            0.0059,
            2.6054,
            2.5988,
            -1.4515,
            -1.4478,
            -0.0065,
            0.0052,
            1.5670,
            -1.5635,
            -1.1428,
            1.1610,
            0.0087,
            0.0087,
            0.0087,
            0.0087,
        ],
    }
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    # start vrsys
    vrsys = OVXRSystem(
        robot=env.robots[0],
        show_control_marker=True,
        system="SteamVR",
        eef_tracking_mode="controller",
        align_anchor_to="camera",
    )
    vrsys.start()

    for _ in range(3000):
        # update the VR system
        vrsys.update()
        # get the action from the VR system and step the environment
        env.step(vrsys.get_robot_teleop_action())

    print("Cleaning up...")
    vrsys.stop()
    og.clear()


if __name__ == "__main__":
    main()
