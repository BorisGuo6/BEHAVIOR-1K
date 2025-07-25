import torch as th
import math
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class CameraMotion(ABC):
    """
    Base class for camera motions during video replay.
    """
    
    def __init__(self, start_frame: int, end_frame: int, **kwargs):
        """
        Args:
            start_frame: Frame number when this motion starts
            end_frame: Frame number when this motion ends
        """
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.duration = end_frame - start_frame
        
    def is_active(self, frame: int) -> bool:
        """Check if this motion should be active at the given frame."""
        return self.start_frame <= frame <= self.end_frame
    
    def get_progress(self, frame: int) -> float:
        """Get progress through the motion (0.0 to 1.0)."""
        if not self.is_active(frame):
            return 0.0
        return (frame - self.start_frame) / max(1, self.duration)
    
    @abstractmethod
    def get_camera_pose(self, frame: int, initial_position: th.Tensor, initial_orientation: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get camera position and orientation for the given frame.
        
        Args:
            frame: Current frame number
            initial_position: Initial camera position
            initial_orientation: Initial camera orientation
            
        Returns:
            (position, orientation) tuple
        """
        pass


class OrbitMotion(CameraMotion):
    """
    Orbit camera around a target point.
    """
    
    def __init__(self, start_frame: int, end_frame: int, 
                 center_point: List[float], radius: float = 3.0, 
                 start_angle: float = 0.0, end_angle: float = 360.0, 
                 height: Optional[float] = None, look_at_center: bool = True,
                 **kwargs):
        """
        Args:
            center_point: [x, y, z] point to orbit around
            radius: Distance from center point
            start_angle: Starting angle in degrees
            end_angle: Ending angle in degrees
            height: Fixed height for camera (if None, uses center_point[2] + radius/2)
            look_at_center: Whether camera should always look at center point
        """
        super().__init__(start_frame, end_frame)
        self.center_point = th.tensor(center_point, dtype=th.float32)
        self.radius = radius
        self.start_angle = math.radians(start_angle)
        self.end_angle = math.radians(end_angle)
        self.height = height if height is not None else center_point[2] + radius / 2
        self.look_at_center = look_at_center
    
    def get_camera_pose(self, frame: int, initial_position: th.Tensor, initial_orientation: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if not self.is_active(frame):
            return initial_position, initial_orientation
            
        progress = self.get_progress(frame)
        angle = self.start_angle + progress * (self.end_angle - self.start_angle)
        
        # Calculate position
        x = self.center_point[0] + self.radius * math.cos(angle)
        y = self.center_point[1] + self.radius * math.sin(angle)
        z = self.height
        position = th.tensor([x, y, z], dtype=th.float32)
        
        if self.look_at_center:
            # Calculate orientation to look at center
            direction = self.center_point - position
            direction = direction / th.norm(direction)
            
            # Create rotation matrix to look at target
            up = th.tensor([0, 0, 1], dtype=th.float32)
            right = th.cross(direction, up)
            right = right / th.norm(right)
            up = th.cross(right, direction)
            
            # Convert to quaternion (simplified - assumes camera looks down -Z)
            # This is a basic implementation; you might want to use a more robust look-at function
            import omnigibson.utils.transform_utils as T
            look_at_matrix = th.stack([right, up, -direction], dim=1)
            orientation = T.mat2quat(look_at_matrix)
        else:
            orientation = initial_orientation
            
        return position, orientation


class ZoomMotion(CameraMotion):
    """
    Zoom camera from one position to another while optionally tracking a target.
    """
    
    def __init__(self, start_frame: int, end_frame: int,
                 start_position: List[float], end_position: List[float],
                 target: Optional[List[float]] = None,
                 interpolation: str = "linear", **kwargs):
        """
        Args:
            start_position: [x, y, z] starting position
            end_position: [x, y, z] ending position  
            target: [x, y, z] point to look at (if None, maintains original orientation)
            interpolation: "linear" or "smooth" (ease-in-out)
        """
        super().__init__(start_frame, end_frame)
        self.start_position = th.tensor(start_position, dtype=th.float32)
        self.end_position = th.tensor(end_position, dtype=th.float32)
        self.target = th.tensor(target, dtype=th.float32) if target else None
        self.interpolation = interpolation
    
    def get_camera_pose(self, frame: int, initial_position: th.Tensor, initial_orientation: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if not self.is_active(frame):
            return initial_position, initial_orientation
            
        progress = self.get_progress(frame)
        
        # Apply interpolation
        if self.interpolation == "smooth":
            # Ease-in-out interpolation
            progress = progress * progress * (3.0 - 2.0 * progress)
        
        # Interpolate position
        position = self.start_position + progress * (self.end_position - self.start_position)
        
        if self.target is not None:
            # Look at target
            direction = self.target - position
            direction = direction / th.norm(direction)
            
            up = th.tensor([0, 0, 1], dtype=th.float32)
            right = th.cross(direction, up)
            right = right / th.norm(right)
            up = th.cross(right, direction)
            
            import omnigibson.utils.transform_utils as T
            look_at_matrix = th.stack([right, up, -direction], dim=1)
            orientation = T.mat2quat(look_at_matrix)
        else:
            orientation = initial_orientation
            
        return position, orientation


class PanMotion(CameraMotion):
    """
    Move camera linearly between two poses while maintaining constant orientation.
    This creates a horizontal pan with no orientation change.
    """
    
    def __init__(self, start_frame: int, 
                 start_position: List[float], end_position: List[float],
                 orientation: List[float], speed: float = 0.5, fps: float = 30.0,
                 interpolation: str = "linear", **kwargs):
        """
        Args:
            start_frame: Frame when motion starts
            start_position: [x, y, z] starting camera position
            end_position: [x, y, z] ending camera position
            orientation: [x, y, z, w] fixed orientation quaternion for entire motion
            speed: Units per second (like traverse_hsf.py)
            fps: Frames per second (used to convert duration to frames)
            interpolation: "linear" or "smooth"
        """
        self.start_position = th.tensor(start_position, dtype=th.float32)
        self.end_position = th.tensor(end_position, dtype=th.float32)
        self.orientation = th.tensor(orientation, dtype=th.float32)
        self.speed = speed
        self.fps = fps
        self.interpolation = interpolation
        
        # Calculate distance and end_frame based on speed (units per second)
        distance = th.norm(self.end_position - self.start_position).item()
        duration_seconds = distance / speed if speed > 0 else 1.0
        duration_frames = int(duration_seconds * fps)
        end_frame = start_frame + duration_frames
        
        super().__init__(start_frame, end_frame)
    
    def get_camera_pose(self, frame: int, initial_position: th.Tensor, initial_orientation: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if not self.is_active(frame):
            return initial_position, initial_orientation
            
        progress = self.get_progress(frame)
        
        if self.interpolation == "smooth":
            progress = progress * progress * (3.0 - 2.0 * progress)
        
        # Interpolate position linearly
        position = self.start_position + progress * (self.end_position - self.start_position)
        
        # Use fixed orientation throughout the motion
        return position, self.orientation


class StaticMotion(CameraMotion):
    """
    Keep camera at a fixed position and orientation.
    """
    
    def __init__(self, start_frame: int, end_frame: int,
                 position: List[float], orientation: List[float], **kwargs):
        """
        Args:
            position: [x, y, z] fixed position
            orientation: [x, y, z, w] fixed orientation quaternion
        """
        super().__init__(start_frame, end_frame)
        self.position = th.tensor(position, dtype=th.float32)
        self.orientation = th.tensor(orientation, dtype=th.float32)
    
    def get_camera_pose(self, frame: int, initial_position: th.Tensor, initial_orientation: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if not self.is_active(frame):
            return initial_position, initial_orientation
        return self.position, self.orientation


class CameraMotionController:
    """
    Controls camera motion during video replay based on a timeline configuration.
    """
    
    MOTION_TYPES = {
        "orbit": OrbitMotion,
        "zoom": ZoomMotion, 
        "pan": PanMotion,
        "static": StaticMotion,
    }
    
    def __init__(self, config_path: Optional[str] = None, motions: Optional[List[Dict]] = None):
        """
        Args:
            config_path: Path to JSON/YAML config file
            motions: List of motion configurations (alternative to config_path)
        """
        self.motions: List[CameraMotion] = []
        
        if config_path:
            self.load_config(config_path)
        elif motions:
            self.load_motions(motions)
    
    def load_config(self, config_path: str):
        """Load camera motions from a configuration file."""
        path = Path(config_path)
        
        if path.suffix.lower() == '.json':
            with open(path) as f:
                config = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(path) as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        self.load_motions(config.get("camera_motions", []))
    
    def load_motions(self, motions_config: List[Dict]):
        """Load camera motions from a list of motion configurations."""
        self.motions = []
        
        for motion_cfg in motions_config:
            motion_type = motion_cfg.get("type")
            if motion_type not in self.MOTION_TYPES:
                raise ValueError(f"Unknown motion type: {motion_type}")
                
            motion_class = self.MOTION_TYPES[motion_type]
            motion = motion_class(**motion_cfg)
            self.motions.append(motion)
        
        # Sort motions by start frame
        self.motions.sort(key=lambda m: m.start_frame)
    
    def get_camera_pose(self, frame: int, initial_position: th.Tensor, initial_orientation: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the camera pose for the given frame.
        
        Args:
            frame: Current frame number
            initial_position: Default camera position
            initial_orientation: Default camera orientation
            
        Returns:
            (position, orientation) tuple
        """
        # Find the active motion for this frame
        active_motion = None
        for motion in self.motions:
            if motion.is_active(frame):
                active_motion = motion
                break
        
        if active_motion:
            return active_motion.get_camera_pose(frame, initial_position, initial_orientation)
        else:
            return initial_position, initial_orientation
    
    def create_example_config(self, output_path: str, total_frames: int = 1000):
        """Create an example configuration file."""
        example_config = {
            "camera_motions": [
                {
                    "type": "static",
                    "start_frame": 0,
                    "end_frame": 100,
                    "position": [-2.0, 0.0, 2.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0]
                },
                {
                    "type": "orbit", 
                    "start_frame": 100,
                    "end_frame": 300,
                    "center_point": [0.0, 0.0, 1.0],
                    "radius": 3.0,
                    "start_angle": 0,
                    "end_angle": 180,
                    "height": 2.5,
                    "look_at_center": True
                },
                {
                    "type": "zoom",
                    "start_frame": 300,
                    "end_frame": 450,
                    "start_position": [3.0, 0.0, 2.0],
                    "end_position": [1.5, 0.0, 1.5],
                    "target": [0.0, 0.0, 1.0],
                    "interpolation": "smooth"
                },
                {
                    "type": "pan",
                    "start_frame": 450,
                    "start_position": [0.0, 0.0, 1.0],
                    "end_position": [2.0, 2.0, 1.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "speed": 0.5,
                    "fps": 30.0,
                    "interpolation": "smooth"
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(example_config, f, indent=2)
        
        print(f"Example camera motion config saved to: {output_path}") 