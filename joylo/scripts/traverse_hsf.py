import numpy as np
from typing import List, Tuple, Dict
import math
import torch as th
import imageio

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardEventHandler

gm.DEFAULT_VIEWER_WIDTH = 3840
gm.DEFAULT_VIEWER_HEIGHT = 2160

class CameraInterpolator:
    def __init__(self, waypoints: List[Dict], target_speed: float = 2.0):
        """
        Initialize camera interpolator with waypoints and target speed.
        
        Args:
            waypoints: List of waypoint dictionaries with 'name', 'pos', and 'ori'
            target_speed: Target camera speed in units per second
        """
        self.waypoints = waypoints
        self.target_speed = target_speed
        self.positions = np.array([wp['pos'] for wp in waypoints])
        self.orientations = np.array([wp['ori'] for wp in waypoints])
        
        # Calculate segment distances and cumulative distances
        self.segment_distances = self._calculate_segment_distances()
        self.total_distance = np.sum(self.segment_distances)
        self.cumulative_distances = np.concatenate([[0], np.cumsum(self.segment_distances)])
        self.total_time = self.total_distance / self.target_speed
        
        # Calculate time at each waypoint
        self.waypoint_times = self.cumulative_distances / self.target_speed
        
        print(f"Total path distance: {self.total_distance:.2f} units")
        print(f"Total travel time: {self.total_time:.2f} seconds")
        print(f"Waypoint times: {self.waypoint_times}")
    
    def _calculate_segment_distances(self) -> np.ndarray:
        """Calculate Euclidean distances between consecutive waypoints."""
        distances = []
        for i in range(len(self.positions) - 1):
            dist = np.linalg.norm(self.positions[i + 1] - self.positions[i])
            distances.append(dist)
        return np.array(distances)
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation for quaternions.
        
        Args:
            q1, q2: Quaternions as numpy arrays [x, y, z, w]
            t: Interpolation parameter [0, 1]
        """
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If dot product is negative, use -q2 to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # Calculate angle between quaternions
        theta_0 = math.acos(abs(dot))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def _find_segment_and_local_t(self, global_t: float) -> Tuple[int, float]:
        """
        Find which segment the global parameter t falls into and the local t within that segment.
        
        Args:
            global_t: Global interpolation parameter [0, 1]
            
        Returns:
            segment_idx: Index of the segment
            local_t: Local interpolation parameter within the segment [0, 1]
        """
        if global_t <= 0:
            return 0, 0.0
        if global_t >= 1:
            return len(self.segment_distances) - 1, 1.0
        
        # Find current distance along path
        current_distance = global_t * self.total_distance
        
        # Find which segment we're in
        segment_idx = np.searchsorted(self.cumulative_distances[1:], current_distance)
        
        # Calculate local t within the segment
        segment_start_dist = self.cumulative_distances[segment_idx]
        segment_length = self.segment_distances[segment_idx]
        
        if segment_length > 0:
            local_t = (current_distance - segment_start_dist) / segment_length
        else:
            local_t = 0.0
        
        return segment_idx, local_t
    
    def interpolate_at_time(self, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get interpolated camera pose at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            position: 3D position as numpy array
            orientation: Quaternion as numpy array [x, y, z, w]
        """
        # Convert time to global parameter
        global_t = min(time / self.total_time, 1.0) if self.total_time > 0 else 0.0
        return self.interpolate(global_t)
    
    def interpolate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate camera pose at parameter t with constant speed.
        
        Args:
            t: Interpolation parameter [0, 1], where 0 is start and 1 is end
            
        Returns:
            position: 3D position as numpy array
            orientation: Quaternion as numpy array [x, y, z, w]
        """
        segment_idx, local_t = self._find_segment_and_local_t(t)
        
        # Get start and end poses for this segment
        start_pos = self.positions[segment_idx]
        end_pos = self.positions[segment_idx + 1]
        start_ori = self.orientations[segment_idx]
        end_ori = self.orientations[segment_idx + 1]
        
        # Linear interpolation for position
        position = start_pos + local_t * (end_pos - start_pos)
        
        # Spherical linear interpolation for orientation
        orientation = self._slerp(start_ori, end_ori, local_t)
        
        return position, orientation
    
    def generate_trajectory(self, fps: float = 30.0) -> List[Dict]:
        """
        Generate a complete trajectory with the specified frame rate.
        
        Args:
            fps: Frames per second
            
        Returns:
            List of dictionaries with 'time', 'position', and 'orientation'
        """
        dt = 1.0 / fps
        trajectory = []
        
        current_time = 0.0
        while current_time <= self.total_time:
            pos, ori = self.interpolate_at_time(current_time)
            trajectory.append({
                'time': current_time,
                'position': pos,
                'orientation': ori
            })
            current_time += dt
        
        # Ensure we include the final waypoint
        if trajectory[-1]['time'] < self.total_time:
            pos, ori = self.interpolate_at_time(self.total_time)
            trajectory.append({
                'time': self.total_time,
                'position': pos,
                'orientation': ori
            })
        
        return trajectory

# Your waypoint data
waypoints_data = [
    {"name": "kitchen", "pos": [7.220621109008789, 3.3456919193267822, 1.9912065267562866], "ori": [0.12101273983716965, 0.6048030257225037, 0.7718288898468018, 0.1544322967529297]},
    {"name": "living room 1", "pos": [9.067486763000488, 3.808771848678589, 1.9579041004180908], "ori": [-0.217719167470932, 0.5649743676185608, 0.7426320910453796, -0.28618156909942627]},
    {"name": "living room 2", "pos": [16.012439727783203, 3.4376437664031982, 1.3308689594268799], "ori": [-0.47005873918533325, 0.4836804270744324, 0.529464602470398, -0.514553427696228]},
    {"name": "utility room", "pos": [22.3378849029541, 3.8088228702545166, 1.4146543741226196], "ori": [-0.09348412603139877, 0.6510287523269653, 0.7456264495849609, -0.10706774890422821]},
    {"name": "outside utility room", "pos": [27.458051681518555, 3.1804323196411133, 1.5083001852035522], "ori": [0.4677571654319763, 0.4838102161884308, 0.5317828059196472, 0.5141381025314331]},
    {"name": "child room 1", "pos": [27.331565856933594, 6.928299903869629, 1.5083001852035522], "ori": [0.4677571952342987, 0.483810156583786, 0.5317828059196472, 0.5141381025314331]},
    {"name": "bathroom 1", "pos": [27.19664764404297, 10.926026344299316, 1.5083001852035522], "ori": [0.4677571952342987, 0.483810156583786, 0.5317828059196472, 0.5141381025314331]},
    {"name": "child room 2", "pos": [27.103891372680664, 13.674463272094727, 1.5083001852035522], "ori": [0.4677571952342987, 0.483810156583786, 0.5317828059196472, 0.5141381025314331]},
    {"name": "child room 3", "pos": [26.960540771484375, 17.922040939331055, 1.5083001852035522], "ori": [0.4677571952342987, 0.483810156583786, 0.5317828059196472, 0.5141381025314331]},
    {"name": "bathroom 2", "pos": [26.86778450012207, 20.670467376708984, 1.5083001852035522], "ori": [0.4677571952342987, 0.483810156583786, 0.5317828059196472, 0.5141381025314331]},
    {"name": "masterbedroom", "pos": [26.7665958404541, 23.668750762939453, 1.5083001852035522], "ori": [0.4677571952342987, 0.483810156583786, 0.5317828059196472, 0.5141381025314331]},
    {"name": "garden", "pos": [26.564218521118164, 29.66531753540039, 1.5083001852035522], "ori": [0.4677571952342987, 0.483810156583786, 0.5317828059196472, 0.5141381025314331]}
]

OBJECTS_TO_HIDE = [
    "sliding_door_tprpvb_7",
    "wall_clock_oqmbtp_0",
    "walls_baibaz_0",
    "walls_cjxcfc_0",
    "walls_fxeanq_0",
    "walls_gjrlnv_0",
    "walls_guxxwh_0",
    "walls_lbxvwa_0",
    "walls_ltqrqp_0",
    "walls_mianco_0",
    "walls_soubdp_0",
    "walls_uplnmr_0",
    "walls_zfhsih_0",
    "tree_rrhqpw_11",
    "fixed_window_bvqijp_0",
    "door_kwbnhy_2",
    "fixed_window_tspbac_2",
    "showerhead_kdvrbf_0",
    "showerhead_kdvrbf_0",
    "fixed_window_tspbac_1",
    "door_kwbnhy_3",
    "door_kwbnhy_0",
    "fixed_window_tspbac_0",
    "fixed_window_usynui_1",
    "fixed_window_usynui_2",
    "fixed_window_usynui_0",
]


cfg = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "house_single_floor",
    },
}

# Load the environment
env = og.Environment(configs=cfg)

for obj in env.scene.objects:
    if obj.name in OBJECTS_TO_HIDE:
        obj.visible = False


KeyboardEventHandler.add_keyboard_callback(
    key=lazy.carb.input.KeyboardInput.ESCAPE,
    callback_fn=lambda: og.shutdown(),
)

KeyboardEventHandler.add_keyboard_callback(
    key=lazy.carb.input.KeyboardInput.SPACE,
    callback_fn=lambda: breakpoint(),
)

for _ in range(100):
    og.sim.render()

interpolator = CameraInterpolator(waypoints_data, target_speed=0.5)
trajectory = interpolator.generate_trajectory(fps=30.0)

# create a video writer
def create_video_writer(fpath, fps=30):
    """
    Creates a video writer to write video frames to when playing back the dataset

    Args:
        fpath (str): Absolute path that the generated video writer will write to. Should end in .mp4
        fps (int): Desired frames per second when generating video

    Returns:
        imageio.Writer: Generated video writer
    """
    assert fpath.endswith(".mp4"), f"Video writer fpath must end with .mp4! Got: {fpath}"
    return imageio.get_writer(fpath, fps=fps)

import time
video_writer = create_video_writer(fpath=f"/home/yhang/BEHAVIOR-1K/joylo/data/filming/traverse_hsf_{time.time()}.mp4")

for waypoint in trajectory:
    og.sim.viewer_camera.set_position_orientation(th.tensor(waypoint["position"], dtype=th.float32), th.tensor(waypoint["orientation"], dtype=th.float32))
    og.sim.step()
    for _ in range(2):
        og.sim.render()
    video_writer.append_data(og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].numpy())

video_writer.close()

og.shutdown()