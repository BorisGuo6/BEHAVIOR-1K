from omnigibson.envs import DataPlaybackWrapper
import torch as th
import os
import omnigibson as og
from omnigibson.macros import gm
import argparse
import sys
from gello.robots.sim_robot.og_teleop_utils import optimize_sim_settings

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160
APERTURE = 35.0

CAMERA_RELATIVE_PRIM_PATH = "/camera" # "/controllable__r1pro__robot_r1/base_link/camera"
CAMERA_OFFSET_POSITION = [ 5.8433, -3.4791,  1.6609] # [-0.4, 0, 2.0]
CAMERA_OFFSET_ORIENTATION = [0.6366, 0.0623, 0.0749, 0.7650] # [0.2706, -0.2706, -0.6533,  0.6533]

# TODO: robot left_eef_link and right_eef_link spheres should not be visible


def replay_hdf5_file(hdf_input_path):
    """
    Replays a single HDF5 file and saves videos to a new folder
    
    Args:
        hdf_input_path: Path to the HDF5 file to replay
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
        "name": "camera",
        "relative_prim_path": CAMERA_RELATIVE_PRIM_PATH,
        "modalities": ["rgb"],
        "sensor_kwargs": {
            "image_height": CAMERA_HEIGHT,
            "image_width": CAMERA_WIDTH,
            "horizontal_aperture": APERTURE,
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

    # Create a list to store video writers and RGB keys
    video_writers = []
    video_rgb_keys = []
    
    # Create video writers for external cameras
    video_writers.append(env.create_video_writer(fpath=f"{video_dir}/camera.mp4"))
    video_rgb_keys.append(f"external::camera::rgb")
    
    # Playback the dataset with all video writers
    for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
        env.playback_episode(
            episode_id=episode_id,
            record_data=False,
            video_writers=video_writers,
            video_rgb_keys=video_rgb_keys,
        )
    
    # Close all video writers
    for writer in video_writers:
        writer.close()

    env.save_data()

    # Always clear the environment to free resources
    og.clear()
        
    print(f"Successfully processed {hdf_input_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to process")
    
    args = parser.parse_args()
    
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
            
        replay_hdf5_file(hdf_file)

    og.shutdown()


if __name__ == "__main__":
    main()
