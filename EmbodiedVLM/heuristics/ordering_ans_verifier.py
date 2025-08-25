"""
This script is the class definition for the online verifier for ordering task.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import scipy.stats
import numpy as np

from EmbodiedVLM.utils.qa_gen_utils import TaskData
from EmbodiedVLM.utils.state_change_translator import StateChangeTranslator
from EmbodiedVLM.utils.scene_graph_utils import SceneGraphReader


class OrderingAnswerVerifier:
    """
    This class is the class definition for the online verifier for ordering task.
    Given the shuffled frame id sequence, read the correct sequence idx, read the input sequence idx, read the task name,
    verfier if the input sequence is correct.
    An input sequence is correct if:
    1. The input sequence is the same as the correct sequence.
    2. The input sequence can reflect the correct sequence, that is, the diffs of the correct sequence is an subset of the diffs of the input sequence.
    3. NEW: The input sequence has high correlation (Kendall's Tau) or high LCS similarity with the correct sequence.
    """
    
    def __init__(
        self,
        input_root_dir: str,
        raw_data_dir: str,
        task_name: str
    ):
        self.task_name = task_name
        self.input_root_dir = Path(input_root_dir)
        self.raw_data_dir = Path(raw_data_dir)

        segmented_scene_graph_file = self.input_root_dir / self.task_name / "segmented_scene_graph_0.json"
        raw_data_task_dir = self.raw_data_dir / self.task_name
        raw_scene_graph_file = raw_data_task_dir / "scene_graph_0.json"

        if not segmented_scene_graph_file.exists():
            raise FileNotFoundError(f"Segmented scene graph file not found: {segmented_scene_graph_file}")
        if not raw_scene_graph_file.exists():
            raise FileNotFoundError(f"Raw scene graph file not found: {raw_scene_graph_file}")
        
        try:
            # Load scene graph data using SceneGraphReader
            segmented_scene_graph_reader = SceneGraphReader(str(segmented_scene_graph_file))
            key_frame_ids = segmented_scene_graph_reader.get_available_frame_ids()

            raw_scene_graph_reader = SceneGraphReader(str(raw_scene_graph_file)) # real working scene graph reader

            # Collect image paths for each frame and sensor
            image_root_path, image_paths = self._collect_image_paths(raw_data_task_dir) # real working image paths

            task_data = TaskData(
                task_name=task_name,
                scene_graph_reader=raw_scene_graph_reader,
                key_frame_ids=key_frame_ids,
                image_paths=image_paths,
                task_dir=str(raw_data_task_dir),
                image_root_path=image_root_path
            )

            print(f"Loaded task: {task_name} with {len(key_frame_ids)} key frames")
        except Exception as e:
            print(f"Error loading task {task_name}: {str(e)}")
            raise e
        
        self.task_data = task_data
        self.translator = StateChangeTranslator(type='forward_dynamics')

    @property
    def sensor_names(self) -> List[str]:
        """
        Extract sensor names from the image_paths dictionary.
        """
        if not self.task_data.image_paths:
            return []
        # Get sensor names from the first frame's image paths
        first_frame_id = list(self.task_data.image_paths.keys())[0]
        return list(self.task_data.image_paths[first_frame_id].keys())

    def _collect_image_paths(self, task_dir: Path, key_frame_ids=None) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        Collect image paths for all key frames and sensors.
        
        Args:
            task_dir (Path): Path to the task directory
            key_frame_ids (List[str]): List of key frame IDs
            
        Returns:
            Dict[str, Dict[str, str]]: Mapping from frame_id to {sensor_name: image_path}
        """
        image_paths = {}
        
        # Find all sensor directories
        sensor_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('external_sensor')]

        image_root_path = task_dir.parent # file structure: image_root/task_name/sensor_name/frame_id.png

        if key_frame_ids is None:
            # Collect all available frame IDs from sensor directories
            all_frame_ids = set()
            for sensor_dir in sensor_dirs:
                # Get all PNG files in this sensor directory
                for image_file in sensor_dir.glob("*.png"):
                    # Extract frame ID from filename (remove .png extension)
                    frame_id = image_file.stem
                    # Convert to int and back to string to ensure consistent formatting
                    try:
                        frame_id_int = int(frame_id)
                        all_frame_ids.add(str(frame_id_int))
                    except ValueError:
                        # Skip files that don't have numeric names
                        continue
            
            # Convert to sorted list for consistent ordering
            key_frame_ids = sorted(all_frame_ids, key=int)
        
        for frame_id in key_frame_ids:
            image_paths[frame_id] = {}
            
            for sensor_dir in sensor_dirs:
                sensor_name = sensor_dir.name
                # Frame files are named with 5-digit zero-padding (e.g., 00051.png)
                image_file = sensor_dir / f"{int(frame_id):05d}.png"
                
                if image_file.exists():
                    image_paths[frame_id][sensor_name] = str(image_file)
        
        return image_root_path, image_paths

    def _translate_sequence_to_signatures(
        self,
        sequence: List[str]
    ) -> List[Set[str]]:
        """
        Translate a sequence of frame IDs into a sequence of signatures.
        """
        signatures = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = self.task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
            )
            signature = self.translator.translate_diff_into_signatures(diff)
            signatures.append(signature)
        return signatures

    def _calculate_kendall_tau(self, correct_sequence: List[int], input_sequence: List[int]) -> float:
        """
        Calculate Kendall's Tau correlation coefficient between two sequences.
        
        Args:
            correct_sequence: Ground truth sequence
            input_sequence: Predicted sequence
            
        Returns:
            Kendall's Tau value (-1 to 1)
        """
        if len(correct_sequence) != len(input_sequence):
            return 0.0
        
        tau, _ = scipy.stats.kendalltau(correct_sequence, input_sequence)
        return tau if not np.isnan(tau) else 0.0

    def _calculate_normalized_lcs(self, correct_sequence: List[int], input_sequence: List[int]) -> float:
        """
        Calculate normalized Longest Common Subsequence (LCS) between two sequences.
        
        Args:
            correct_sequence: Ground truth sequence
            input_sequence: Predicted sequence
            
        Returns:
            Normalized LCS score (0 to 1)
        """
        m, n = len(correct_sequence), len(input_sequence)
        if m == 0 or n == 0:
            return 0.0
        
        # Dynamic programming table for LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if correct_sequence[i - 1] == input_sequence[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # Normalize by the maximum possible length
        return dp[m][n] / max(m, n)
    
    def verify(
        self,
        shuffled_frame_id_sequence: List[str],
        correct_state_idx_sequence: List[int],
        input_state_idx_sequence: List[int]
    ) -> Dict[str, Any]:
        """
        Verify if the input sequence is correct using multiple evaluation methods.
        
        Args:
            shuffled_frame_id_sequence: The shuffled frame sequence for this specific question
            correct_state_idx_sequence: Ground truth answer (1-indexed)
            input_state_idx_sequence: Predicted answer (1-indexed)
            
        Returns:
            Dictionary containing evaluation results with raw scores
        """
        # Convert to 0-indexed
        correct_sequence = [idx - 1 for idx in correct_state_idx_sequence]
        input_sequence = [idx - 1 for idx in input_state_idx_sequence]

        # Calculate exact match
        exact_match = correct_sequence == input_sequence
        
        # Calculate semantic match
        semantic_match = False
        if len(correct_sequence) == len(input_sequence):
            try:
                # Convert indices to frame sequences
                correct_frame_id_sequence = [shuffled_frame_id_sequence[idx] for idx in correct_sequence]
                input_frame_id_sequence = [shuffled_frame_id_sequence[idx] for idx in input_sequence]

                # Get signatures for both sequences
                correct_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence)
                input_signatures = self._translate_sequence_to_signatures(input_frame_id_sequence)

                # Check if all correct signatures are subsets of input signatures
                semantic_match = True
                for i in range(len(correct_signatures)):
                    if not correct_signatures[i].issubset(input_signatures[i]):
                        semantic_match = False
                        break
            except Exception as e:
                semantic_match = False

        # Combined match result: True if either exact OR semantic match
        match = exact_match or semantic_match

        # Calculate correlation metrics (raw scores, no thresholding)
        kendall_tau = self._calculate_kendall_tau(correct_sequence, input_sequence)
        normalized_lcs = self._calculate_normalized_lcs(correct_sequence, input_sequence)

        results = {
            'exact_match': exact_match,
            'semantic_match': semantic_match,
            'match': match,  # Combined match result
            'kendall_tau': kendall_tau,  # Raw score
            'normalized_lcs': normalized_lcs  # Raw score
        }

        return results

    def verify_forward(
        self,
        shuffled_frame_id_sequence: List[str],
        correct_state_idx_sequence: List[int],
        input_state_idx_sequence: List[int]
    ) -> Dict[str, Any]:
        """
        Verify forward dynamics ordering - alias for verify method.
        """
        return self.verify(shuffled_frame_id_sequence, correct_state_idx_sequence, input_state_idx_sequence)

    def verify_inverse(
        self,
        shuffled_frame_id_sequence: List[str],
        correct_state_idx_sequence: List[int],
        input_state_idx_sequence: List[int]
    ) -> Dict[str, Any]:
        """
        Verify inverse dynamics ordering - currently same as forward dynamics.
        In the future, this could implement different logic for inverse dynamics.
        """
        return self.verify(shuffled_frame_id_sequence, correct_state_idx_sequence, input_state_idx_sequence)
        