from omnigibson.envs import DataPlaybackWrapper
import torch as th
import os
import omnigibson as og
from omnigibson.macros import gm
import argparse
import sys
from gello.robots.sim_robot.og_teleop_utils import optimize_sim_settings
from camera_motions import CameraMotionController
from omnigibson.object_states import *

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128


TRIAL_RUN = True
TRAVERSAL_SHOT = False

CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160
APERTURE = 35.0

CAMERA_RELATIVE_PRIM_PATH = "/main_camera" # "/controllable__r1pro__robot_r1/base_link/main_camera"
CAMERA_OFFSET_POSITION = [20.0, 10.0, 1.0] # [ 5.8433, -3.4791,  1.6609] # [-0.4, 0, 2.0]
CAMERA_OFFSET_ORIENTATION = [0.0, 0.0, 0.0, 1.0] # [0.6366, 0.0623, 0.0749, 0.7650] # [0.2706, -0.2706, -0.6533,  0.6533]

OBJECTS_TO_HIDE = [
    "clothesline_pole_smrqrw_0",
    "clothesline_pole_smrqrw_1",
    "door_kwbnhy_0",
    "door_kwbnhy_2",
    "door_kwbnhy_3",
    "fixed_window_tspbac_0",
    "fixed_window_tspbac_1",
    "fixed_window_tspbac_2",
    "fixed_window_usynui_0",
    "fixed_window_usynui_2",
    "fixed_window_usynui_1",
    # "sliding_door_tprpvb_0",
    # "sliding_door_tprpvb_1",
    # "sliding_door_tprpvb_11",
    # "sliding_door_tprpvb_6",
    "wall_clock_oqmbtp_0",
    "walls_baibaz_0",
    "walls_cjxcfc_0",
    "walls_fxeanq_0",
    "walls_gjrlnv_0",
    "walls_guxxwh_0",
    "walls_lbxvwa_0",
    "walls_ltqrqp_0",
    "walls_soubdp_0",
    "walls_zfhsih_0",
    "showerhead_kdvrbf_0",
]

SLIDING_DOORS_TO_OPEN = [
    "sliding_door_tprpvb_0",
    "sliding_door_tprpvb_1",
    "sliding_door_tprpvb_11",
    "sliding_door_tprpvb_6",
]


def replay_hdf5_file(hdf_input_path, camera_motion_config=None):
    """
    Replays a single HDF5 file and saves videos to a new folder
    
    Args:
        hdf_input_path: Path to the HDF5 file to replay
        camera_motion_config: Path to camera motion config file or None for static camera
    """
    # Create folder with same name as HDF5 file (without extension)
    base_name = os.path.basename(hdf_input_path)
    folder_name = os.path.splitext(base_name)[0]
    folder_path = os.path.join(os.path.dirname(hdf_input_path), folder_name)
    
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Define output paths
    hdf_output_path = os.path.join(folder_path, f"{folder_name}_replay.hdf5")
    video_dir = folder_path
    
    # This flag is needed to run data playback wrapper
    gm.ENABLE_TRANSITION_RULES = False
    
    camera_config = {
        "sensor_type": "VisionSensor",
        "name": "main_camera",
        "relative_prim_path": CAMERA_RELATIVE_PRIM_PATH,
        "modalities": ["rgb"],
        "sensor_kwargs": {
            "image_height": CAMERA_HEIGHT,
            "image_width": CAMERA_WIDTH,
            # "horizontal_aperture": APERTURE,
        },
        "position": th.tensor(CAMERA_OFFSET_POSITION, dtype=th.float32),
        "orientation": th.tensor(CAMERA_OFFSET_ORIENTATION, dtype=th.float32),
        "pose_frame": "parent",
    }

    # Create the environment
    # TODO: need to merge scene files to create full scene for final rendering
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=hdf_input_path,
        output_path=hdf_output_path,
        robot_obs_modalities=["rgb"],
        robot_sensor_config=None,
        external_sensors_config=[camera_config],
        exclude_sensor_names=["zed", "realsense"],
        n_render_iterations=1,
        only_successes=False,
        include_task=True,
        include_task_obs=False,
        include_robot_control=False,
        include_contacts=False,
    )
    
    robot = env.robots[0]
    robot.links["left_eef_link"].visible = False
    robot.links["right_eef_link"].visible = False

    # Optimize rendering for faster speeds
    og.sim.add_callback_on_play("optimize_rendering", optimize_sim_settings)

    # Initialize camera motion controller if config provided
    camera_motion_controller = None
    if camera_motion_config:
        try:
            camera_motion_controller = CameraMotionController(config_path=camera_motion_config)
            print(f"Loaded camera motion config from: {camera_motion_config}")
        except Exception as e:
            print(f"Warning: Failed to load camera motion config: {e}")
            print("Continuing with static camera...")

    # Get the external camera sensor for motion control
    external_camera = None
    for sensor_name, sensor in env.external_sensors.items():
        if "main_camera" in sensor_name:
            external_camera = sensor
            break
    
    if external_camera is None:
        print("Warning: Could not find external camera sensor for motion control")

    # Store initial camera pose for motion calculations
    initial_camera_position = th.tensor(CAMERA_OFFSET_POSITION, dtype=th.float32)
    initial_camera_orientation = th.tensor(CAMERA_OFFSET_ORIENTATION, dtype=th.float32)

    # Create a list to store video writers and RGB keys
    video_writers = []
    video_rgb_keys = []
    
    # Create video writers for external cameras
    video_writers.append(env.create_video_writer(fpath=f"{video_dir}/main_camera.mp4"))
    video_rgb_keys.append(f"external::main_camera::rgb")
    
    # Playback the dataset with all video writers
    assert env.input_hdf5["data"].attrs["n_episodes"] == 1, "Only one episode is supported for now"
    
    # Custom playback with camera motion
    playback_episode_with_camera_motion(
        env=env,
        episode_id=0,
        video_writers=video_writers,
        video_rgb_keys=video_rgb_keys,
        camera_motion_controller=camera_motion_controller,
        external_camera=external_camera,
        initial_camera_position=initial_camera_position,
        initial_camera_orientation=initial_camera_orientation
    )
    
    # Close all video writers
    for writer in video_writers:
        writer.close()

    env.save_data()

    # Always clear the environment to free resources
    og.shutdown()
        
    print(f"Successfully processed {hdf_input_path}")


def playback_episode_with_camera_motion(env, episode_id, video_writers, video_rgb_keys, 
                                       camera_motion_controller, external_camera,
                                       initial_camera_position, initial_camera_orientation):
    """
    Custom episode playback with camera motion support.
    
    Args:
        env: DataPlaybackWrapper environment
        episode_id: Episode ID to playback
        video_writers: List of video writers
        video_rgb_keys: List of RGB observation keys
        camera_motion_controller: CameraMotionController instance or None
        external_camera: External camera sensor for motion control
        initial_camera_position: Initial camera position
        initial_camera_orientation: Initial camera orientation
    """
    data_grp = env.input_hdf5["data"]
    assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
    traj_grp = data_grp[f"demo_{episode_id}"]

    # Grab episode data
    try:
        from omnigibson.utils.python_utils import h5py_group_to_torch
        import json
        
        transitions = json.loads(traj_grp.attrs["transitions"])
        traj_grp = h5py_group_to_torch(traj_grp)
        init_metadata = traj_grp["init_metadata"]
        action = traj_grp["action"]
        state = traj_grp["state"]
        state_size = traj_grp["state_size"]
        reward = traj_grp["reward"]
        terminated = traj_grp["terminated"]
        truncated = traj_grp["truncated"]
    except KeyError as e:
        print(f"Got error when trying to load episode {episode_id}:")
        print(f"Error: {str(e)}")
        return

    # Reset environment and update this to be the new initial state
    env.scene.restore(env.scene_file, update_initial_file=True)

    # Reset object attributes from the stored metadata
    with og.sim.stopped():
        for attr, vals in init_metadata.items():
            assert len(vals) == env.scene.n_objects
        for i, obj in enumerate(env.scene.objects):
            for attr, vals in init_metadata.items():
                val = vals[i]
                setattr(obj, attr, val.item() if val.ndim == 0 else val)
    env.reset()

    # Restore to initial state
    og.sim.load_state(state[0, : int(state_size[0])], serialized=True)
    
    if TRAVERSAL_SHOT:
        for obj in OBJECTS_TO_HIDE:
            env.scene.object_registry("name", obj).visible = False
        for door in SLIDING_DOORS_TO_OPEN:
            env.scene.object_registry("name", door).states[Open].set_value(True)

    # Record initial observations with camera motion
    frame_count = 0

    init_obs, _, _, _, init_info = env.env.step(action=action[0], n_render_iterations=env.n_render_iterations)
    step_data = {"obs": env._process_obs(obs=init_obs, info=init_info)}
    env.current_traj_history.append(step_data)

    frame_count += 1
    
    for _ in range(20):
        og.sim.render()

    for i, (a, s, ss, r, te, tr) in enumerate(
        zip(action, state[1:], state_size[1:], reward, terminated, truncated)
    ):
        # Execute any transitions that should occur at this current step
        if str(i) in transitions:
            cur_transitions = transitions[str(i)]
            scene = og.sim.scenes[0]
            for add_sys_name in cur_transitions["systems"]["add"]:
                scene.get_system(add_sys_name, force_init=True)
            for remove_sys_name in cur_transitions["systems"]["remove"]:
                scene.clear_system(remove_sys_name)
            for remove_obj_name in cur_transitions["objects"]["remove"]:
                obj = scene.object_registry("name", remove_obj_name)
                scene.remove_object(obj)
            for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                from omnigibson.utils.python_utils import create_object_from_init_info
                obj = create_object_from_init_info(add_obj_info)
                scene.add_object(obj)
                obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
            # Step physics to initialize any new objects
            og.sim.step()

        if not TRIAL_RUN:
            # Restore the sim state
            og.sim.load_state(s[: int(ss)], serialized=True)

        if camera_motion_controller and frame_count < camera_motion_controller.motions[0].start_frame:
            env.step_count += 1
            frame_count += 1
            continue
        elif camera_motion_controller and frame_count > camera_motion_controller.motions[0].end_frame:
            break
        elif frame_count < 1800:
            env.step_count += 1
            frame_count += 1
            continue
        # elif frame_count > 2800:
        #     break
        else:
            if TRIAL_RUN:
                og.sim.load_state(s[: int(ss)], serialized=True)
            
            # Update camera position for current frame if motion controller exists
            if camera_motion_controller and external_camera:
                position, orientation = camera_motion_controller.get_camera_pose(
                    frame_count, initial_camera_position, initial_camera_orientation
                )
                external_camera.set_position_orientation(position=position, orientation=orientation)
                if frame_count == camera_motion_controller.motions[0].start_frame:
                    og.sim.step()
                    for _ in range(10):
                        og.sim.render()
            
            # Take simulation step with action
            env.current_obs, _, _, _, info = env.env.step(action=a, n_render_iterations=3)

            # Record data
            step_data = env._parse_step_data(
                action=a,
                obs=env.current_obs,
                reward=r,
                terminated=te,
                truncated=tr,
                info=info,
            )
            env.current_traj_history.append(step_data)

            # Write frame to video if needed
            if video_writers is not None:
                for writer, rgb_key in zip(video_writers, video_rgb_keys):
                    from omnigibson.utils.python_utils import assert_valid_key
                    assert_valid_key(rgb_key, env.current_obs.keys(), "video_rgb_key")
                    writer.append_data(env.current_obs[rgb_key][:, :, :3].numpy())

        env.step_count += 1
        frame_count += 1
        print(f"Frame {frame_count} processed")

    env.flush_current_traj()


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos with optional camera motion")
    parser.add_argument("--dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to process")
    parser.add_argument("--camera_motion", help="Path to camera motion configuration file (JSON/YAML)")
    parser.add_argument("--create_example_config", help="Create an example camera motion config file at the specified path")
    
    args = parser.parse_args()
    
    # Create example config if requested
    if args.create_example_config:
        controller = CameraMotionController()
        controller.create_example_config(args.create_example_config)
        print(f"Example camera motion config created at: {args.create_example_config}")
        return
    
    if args.dir and os.path.isdir(args.dir):
        # Process all HDF5 files in the directory (non-recursively)
        hdf_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                    if f.lower().endswith('.hdf5') and os.path.isfile(os.path.join(args.dir, f))]
        
        if not hdf_files:
            print(f"No HDF5 files found in directory: {args.dir}")
        else:
            print(f"Found {len(hdf_files)} HDF5 files to process")
    elif args.files:
        # Process individual files specified
        hdf_files = args.files
    else:
        parser.print_help()
        print("\nError: Either --dir or --files must be specified", file=sys.stderr)
        return
    
    # Process each file
    for hdf_file in hdf_files:
        if not os.path.exists(hdf_file):
            print(f"Error: File {hdf_file} does not exist", file=sys.stderr)
            continue
            
        replay_hdf5_file(hdf_file, camera_motion_config=args.camera_motion)

    og.shutdown()


if __name__ == "__main__":
    main()
