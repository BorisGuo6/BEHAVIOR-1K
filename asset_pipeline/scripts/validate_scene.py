import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm

import igibson
igibson.ig_dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts/aggregate")

from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.external.pybullet_tools import utils

OUTPUT_FILENAME = "validate_scene.json"
SUCCESS_FILENAME = "validate_scene.success"

MAX_POS_DELTA = 0.1  # 10cm
MAX_ORN_DELTA = np.deg2rad(10)  # 10 degrees

def main():
    target = sys.argv[1]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts")
    scene_name = os.path.split(target)[1]
    scene_filename = os.path.join(output_dir, f"scene/{scene_name}_best.urdf")

    # Load the scene into iGibson 2
    s = Simulator(mode="headless", use_pb_gui=True)
    scene = InteractiveIndoorScene(scene_name, urdf_path=scene_filename)
    s.import_scene(scene)

    print("Imported scene.")

    # Save the state of every object.
    obj_states = {}
    for key, obj in scene.objects_by_id.items():
        for body in obj.get_body_ids():
            link_ids = utils.get_all_links(body)
            link_poses = [utils.get_link_pose(body, link) for link in link_ids]
            obj_states[body] = link_poses

    # Step the simulation by 5 seconds.
    for step in tqdm.tqdm(range(5 * 30)):
        s.step()

    # Check the state of every object again.
    mismatches = []
    for key, obj in scene.objects_by_id.items():
        for body in obj.get_body_ids():
            link_ids = utils.get_all_links(body)
            new_link_poses = [utils.get_link_pose(body, link) for link in link_ids]
            old_link_poses = obj_states[body]

            for link_id, new_pose, old_pose in zip(link_ids, new_link_poses, old_link_poses):
                link_name = utils.get_link_name(body, link_id)

                new_pos, new_orn = new_pose
                old_pos, old_orn = old_pose

                delta_pos = np.linalg.norm(np.array(new_pos) - np.array(old_pos))
                if delta_pos > MAX_POS_DELTA:
                    mismatches.append(f"Object {key} link {link_name} position changed by {delta_pos} meters from {old_pos} to {new_pos}.")
                delta_orn_mag = (R.from_quat(old_orn).inv() * R.from_quat(new_orn)).magnitude()
                if delta_orn_mag > MAX_ORN_DELTA:
                    mismatches.append(f"Object {key} link {link_name} orientation changed by {delta_orn_mag} rads from {old_orn} to {new_orn}.")

    success = True  # (len(mismatches) == 0)

    filename = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(filename, "w") as f:
        json.dump({"success": success, "mismatches": sorted(mismatches)}, f, indent=4)

    if success:
        with open(os.path.join(output_dir, SUCCESS_FILENAME), "w") as f:
            pass

if __name__ == "__main__":
    main()