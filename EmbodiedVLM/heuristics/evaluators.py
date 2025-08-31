"""
This script is the class definition for the online verifier for ordering task.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import scipy.stats
import numpy as np
import json
import re
import ast
from collections import defaultdict

from EmbodiedVLM.utils.qa_gen_utils import TaskData
from EmbodiedVLM.utils.state_change_translator import StateChangeTranslator
from EmbodiedVLM.utils.scene_graph_utils import SceneGraphReader


class TaskWiseOrderingVerifier:
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
        task_data: TaskData
    ):
        """
        Initialize TaskWiseOrderingVerifier with pre-created TaskData.
        
        Args:
            task_data (TaskData): Pre-configured TaskData object
        """
        self.task_data = task_data
        self.task_name = task_data.task_name
        self.translator = StateChangeTranslator(type='forward_dynamics')
        
        print(f"Loaded task: {self.task_name} with {len(task_data.key_frame_ids)} key frames")

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

    def _translate_sequence_to_signatures(
        self,
        sequence: List[str],
        partial_diff: bool = True
    ) -> List[Set[str]]:
        """
        Translate a sequence of frame IDs into a sequence of signatures.
        """
        signatures = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = self.task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=partial_diff
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

    def verify_forward(
        self,
        shuffled_frame_id_sequence: List[str],
        correct_state_idx_sequence: List[int],
        input_state_idx_sequence: List[int]
    ) -> Dict[str, Any]:
        """
        Verify if the input sequence is correct for forward dynamics.
        
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
                correct_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence, partial_diff=True)
                input_signatures = self._translate_sequence_to_signatures(input_frame_id_sequence, partial_diff=False)

                # Check if all correct signatures are subsets of input signatures
                semantic_match = True
                for i in range(len(correct_signatures)):
                    # as long as the action sequence(ground truth) can describe the input sequence, it is correct
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


        if match:
            kendall_tau = 1
            normalized_lcs = 1

        results = {
            'exact_match': exact_match,
            'semantic_match': semantic_match,
            'match': match,  # Combined match result
            'kendall_tau': kendall_tau,  # Raw score
            'normalized_lcs': normalized_lcs  # Raw score
        }

        return results

    def verify_inverse(
        self,
        correct_frame_id_sequence: List[str],
        correct_action_idx_sequence: List[int],
        input_action_idx_sequence: List[int]
    ) -> Dict[str, Any]:
        """
        Verify inverse dynamics ordering
        In the future, this could implement different logic for inverse dynamics.
        """
        # Convert to 0-indexed
        correct_sequence = [idx - 1 for idx in correct_action_idx_sequence]
        input_sequence = [idx - 1 for idx in input_action_idx_sequence]

        # Calculate exact match
        exact_match = correct_sequence == input_sequence
        
        # Calculate semantic match for inverse dynamics
        semantic_match = False
        if len(correct_sequence) == len(input_sequence):
            try:
                # Get signatures ordered by the correct action sequence
                correct_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence, partial_diff=True)
                correct_full_signatures = self._translate_sequence_to_signatures(correct_frame_id_sequence, partial_diff=False)
                
                # Example: if signatures are [x, y, z] ordered by correct action sequence
                # and correct action idx sequence is [3, 1, 2] (1-indexed) -> [2, 0, 1] (0-indexed)
                # This means: action 3 comes first, then action 1, then action 2
                # So the shuffled signatures would be ordered as [z, x, y] (following indices [2, 0, 1])
                
                # Create a mapping from correct action indices to signature positions
                # correct_sequence contains the 0-indexed positions in the shuffled order
                shuffled_signatures = [None] * len(correct_signatures)
                for i, action_idx in enumerate(correct_sequence):
                    if 0 <= action_idx < len(correct_signatures):
                        shuffled_signatures[action_idx] = correct_signatures[i]
                
                # Remove any None entries (in case of invalid indices)
                shuffled_signatures = [sig for sig in shuffled_signatures if sig is not None]
                
                # Now get the input signatures based on the input sequence using correct_signatures
                # The input signatures are a rearrangement of correct_signatures according to input_sequence
                input_signatures = [None] * len(shuffled_signatures)
                for i, action_idx in enumerate(input_sequence):
                    if 0 <= action_idx < len(shuffled_signatures):
                        input_signatures[i] = shuffled_signatures[action_idx]
                
                # Remove any None entries (in case of invalid indices)
                input_signatures = [sig for sig in input_signatures if sig is not None]
                
                # Check if all input signatures are subsets of correct signatures
                semantic_match = True
                if len(correct_signatures) != len(input_signatures):
                    semantic_match = False
                else:
                    for i in range(len(input_signatures)): # logic is different from foward dynamics
                        # as long as the input action sequence can describe the states, it is correct
                        # here, we do not force the action to be the visible action
                        if not input_signatures[i].issubset(correct_full_signatures[i]):
                            semantic_match = False
                            break

            except Exception as e:
                semantic_match = False
        
        # Combined match result: True if either exact OR semantic match
        match = exact_match or semantic_match

        # Calculate correlation metrics (raw scores, no thresholding)
        kendall_tau = self._calculate_kendall_tau(correct_sequence, input_sequence)
        normalized_lcs = self._calculate_normalized_lcs(correct_sequence, input_sequence)

        if match:
            kendall_tau = 1
            normalized_lcs = 1

        results = {
            'exact_match': exact_match,
            'semantic_match': semantic_match,
            'match': match,  # Combined match result
            'kendall_tau': kendall_tau,  # Raw score
            'normalized_lcs': normalized_lcs  # Raw score
        }

        return results


class OrderingEvaluator:
    """
    This class is the wrapper class for batch evaluation of ordering tasks.
    It processes JSONL files containing multiple data points and provides
    comprehensive evaluation metrics and reports.
    """
    
    def __init__(
        self,
        input_root_dir: str,
        raw_data_dir: str
    ):
        """
        Initialize the OrderingEvaluator.
        
        Args:
            input_root_dir (str): Root directory containing segmented scene graphs
            raw_data_dir (str): Directory containing raw data and scene graphs
        """
        self.input_root_dir = Path(input_root_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.eval_results = {}  # Cache for evaluation results
        self._verifiers_cache = {}  # Cache for TaskWiseOrderingVerifier instances
        self.skipped_items = []  # Track skipped items with reasons
        
    def _parse_answer_string(self, answer_str: str) -> Optional[List[int]]:
        """
        Extract a Python list from the answer string.
        
        Args:
            answer_str (str): Raw answer string from model output
            
        Returns:
            Optional[List[int]]: Extracted list or None if parsing fails
        """
        if not answer_str or not isinstance(answer_str, str):
            return None
            
        try:
            # Try to find list patterns like [1, 2, 3] or [1,2,3]
            list_pattern = r'\[[\d\s,]+\]'
            matches = re.findall(list_pattern, answer_str)
            
            if matches:
                # Take the last match and evaluate it
                list_str = matches[-1]
                parsed_list = ast.literal_eval(list_str)
                
                # Ensure it's a list of integers
                if isinstance(parsed_list, list) and all(isinstance(x, int) for x in parsed_list):
                    # some may start from 0, we need to convert to 1-indexed
                    if min(parsed_list) == 0:
                        parsed_list = [x + 1 for x in parsed_list]
                    # some may start from 2 or larger, we need to convert to 1-indexed
                    if min(parsed_list) > 1:
                        delta = min(parsed_list) - 1
                        parsed_list = [x - delta for x in parsed_list]
                    return parsed_list
                    
        except (SyntaxError, ValueError, TypeError):
            pass
            
        # Alternative: try to extract numbers separated by commas/spaces
        try:
            # Look for sequences of numbers
            numbers = re.findall(r'\d+', answer_str)
            if numbers:
                return [int(num) for num in numbers]
        except (ValueError, TypeError):
            pass
            
        return None
        
    def _create_task_data(self, task_name: str) -> TaskData:
        """
        Create TaskData object for a specific task.
        
        Args:
            task_name (str): Name of the task
            
        Returns:
            TaskData: Configured TaskData object
            
        Raises:
            FileNotFoundError: If required files are not found
        """
        segmented_scene_graph_file = self.input_root_dir / task_name / "segmented_scene_graph_0.json"
        raw_data_task_dir = self.raw_data_dir / task_name
        raw_scene_graph_file = raw_data_task_dir / "scene_graph_0.json"

        if not segmented_scene_graph_file.exists():
            raise FileNotFoundError(f"Segmented scene graph file not found: {segmented_scene_graph_file}")
        if not raw_scene_graph_file.exists():
            raise FileNotFoundError(f"Raw scene graph file not found: {raw_scene_graph_file}")
        
        # Load scene graph data using SceneGraphReader
        segmented_scene_graph_reader = SceneGraphReader(str(segmented_scene_graph_file))
        key_frame_ids = segmented_scene_graph_reader.get_available_frame_ids()

        raw_scene_graph_reader = SceneGraphReader(str(raw_scene_graph_file))

        # Collect image paths for each frame and sensor
        image_root_path, image_paths = self._collect_image_paths(raw_data_task_dir)

        task_data = TaskData(
            task_name=task_name,
            scene_graph_reader=raw_scene_graph_reader,
            key_frame_ids=key_frame_ids,
            image_paths=image_paths,
            task_dir=str(raw_data_task_dir),
            image_root_path=image_root_path
        )

        return task_data
        
    def _collect_image_paths(self, task_dir: Path, key_frame_ids=None) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        Collect image paths for all key frames and sensors.
        This method is extracted from TaskWiseOrderingVerifier for reuse.
        
        Args:
            task_dir (Path): Path to the task directory
            key_frame_ids (List[str]): List of key frame IDs
            
        Returns:
            Tuple[str, Dict[str, Dict[str, str]]]: Image root path and mapping from frame_id to {sensor_name: image_path}
        """
        image_paths = {}
        
        # Find all sensor directories
        sensor_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('external_sensor')]

        image_root_path = task_dir.parent

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
        
    def _gather_verifiers(self, jsonl_path: str) -> Dict[str, TaskWiseOrderingVerifier]:
        """
        Scan the JSONL file to collect all unique task names and create verifiers.
        
        Args:
            jsonl_path (str): Path to the JSONL file
            
        Returns:
            Dict[str, TaskWiseOrderingVerifier]: Dictionary mapping task names to verifiers
        """
        unique_task_names = set()
        
        # First pass: collect all unique task names
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data_point = json.loads(line.strip())
                    task_name = data_point.get('task_name')
                    if task_name:
                        unique_task_names.add(task_name)
                except json.JSONDecodeError:
                    continue
                    
        # Create verifiers for each unique task
        verifiers = {}
        for task_name in unique_task_names:
            if task_name not in self._verifiers_cache:
                try:
                    # Create TaskData first
                    task_data = self._create_task_data(task_name)
                    # Pass TaskData to TaskWiseOrderingVerifier
                    verifier = TaskWiseOrderingVerifier(task_data=task_data)
                    self._verifiers_cache[task_name] = verifier
                    print(f"Created verifier for task: {task_name}")
                except Exception as e:
                    print(f"Failed to create verifier for task {task_name}: {str(e)}")
                    continue
                    
            verifiers[task_name] = self._verifiers_cache[task_name]
            
        return verifiers
        
    def evaluate(self, jsonl_path: str) -> None:
        """
        Main evaluation method that processes the entire JSONL file.
        
        Args:
            jsonl_path (str): Path to the JSONL evaluation file
        """
        print(f"Starting evaluation of {jsonl_path}")
        
        # Gather all required verifiers
        verifiers = self._gather_verifiers(jsonl_path)
        
        if not verifiers:
            print("No valid verifiers found. Evaluation aborted.")
            return
            
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each data point in the JSONL file
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data_point = json.loads(line.strip())
                    data_id = data_point.get('id')
                    
                    if not data_id:
                        print(f"Line {line_num}: Missing 'id' field, skipping")
                        error_count += 1
                        self.skipped_items.append({
                            'id': 'unknown',
                            'line_num': line_num,
                            'reason': 'missing_id_field',
                            'task_name': data_point.get('task_name', 'unknown'),
                            'task_type': data_point.get('type', 'unknown')
                        })
                        continue
                        
                    # Check if already evaluated
                    if data_id in self.eval_results and 'eval_metrics' in self.eval_results[data_id]:
                        skipped_count += 1
                        self.skipped_items.append({
                            'id': data_id,
                            'line_num': line_num,
                            'reason': 'already_evaluated',
                            'task_name': data_point.get('task_name', 'unknown'),
                            'task_type': data_point.get('type', 'unknown')
                        })
                        continue
                        
                    # Extract required fields
                    task_name = data_point.get('task_name')
                    task_type = data_point.get('type')
                    key_frame_ids = data_point.get('key_frame_ids', [])
                    gt_answer = data_point.get('gt_answer', [])
                    answer = data_point.get('answer', '')
                    
                    # Validate required fields
                    if not task_name or task_name not in verifiers:
                        print(f"Line {line_num}: Invalid or missing task_name '{task_name}', skipping")
                        error_count += 1
                        self.skipped_items.append({
                            'id': data_id,
                            'line_num': line_num,
                            'reason': 'invalid_or_missing_task_name',
                            'task_name': task_name or 'missing',
                            'task_type': task_type
                        })
                        continue
                        
                    if not gt_answer or not key_frame_ids:
                        print(f"Line {line_num}: Missing gt_answer or key_frame_ids, skipping")
                        error_count += 1
                        self.skipped_items.append({
                            'id': data_id,
                            'line_num': line_num,
                            'reason': 'missing_gt_answer_or_key_frames',
                            'task_name': task_name,
                            'task_type': task_type
                        })
                        continue
                        
                    # Parse the model's answer
                    parsed_answer = self._parse_answer_string(answer)
                    if parsed_answer is None:
                        print(f"Line {line_num}: Failed to parse answer '{answer}', skipping")
                        error_count += 1
                        self.skipped_items.append({
                            'id': data_id,
                            'line_num': line_num,
                            'reason': 'failed_to_parse_answer',
                            'task_name': task_name,
                            'task_type': task_type,
                            'raw_answer': answer
                        })
                        continue
                        
                    # Perform verification
                    verifier = verifiers[task_name]
                    
                    # Determine which verification method to use based on task type
                    if task_type and 'inverse' in task_type.lower():
                        eval_result = verifier.verify_inverse(key_frame_ids, gt_answer, parsed_answer)
                    else:
                        eval_result = verifier.verify_forward(key_frame_ids, gt_answer, parsed_answer)
                    
                    # Store the complete result
                    result_entry = {
                        'id': data_id,
                        'type': task_type,
                        'task_name': task_name,
                        'key_frame_ids': key_frame_ids,
                        'gt_answer': gt_answer,
                        'parsed_answer': parsed_answer,
                        'raw_answer': answer,
                        'eval_metrics': eval_result
                    }
                    
                    self.eval_results[data_id] = result_entry
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        print(f"Processed {processed_count} data points...")
                        
                except json.JSONDecodeError:
                    print(f"Line {line_num}: Invalid JSON format, skipping")
                    error_count += 1
                    self.skipped_items.append({
                        'id': 'unknown',
                        'line_num': line_num,
                        'reason': 'invalid_json_format',
                        'task_name': 'unknown',
                        'task_type': 'unknown'
                    })
                    continue
                except Exception as e:
                    print(f"Line {line_num}: Error during evaluation: {str(e)}")
                    error_count += 1
                    # Try to extract some info from the data_point if it exists
                    try:
                        data_point = json.loads(line.strip())
                        self.skipped_items.append({
                            'id': data_point.get('id', 'unknown'),
                            'line_num': line_num,
                            'reason': f'evaluation_error: {str(e)}',
                            'task_name': data_point.get('task_name', 'unknown'),
                            'task_type': data_point.get('type', 'unknown')
                        })
                    except:
                        self.skipped_items.append({
                            'id': 'unknown',
                            'line_num': line_num,
                            'reason': f'evaluation_error: {str(e)}',
                            'task_name': 'unknown',
                            'task_type': 'unknown'
                        })
                    continue
                    
        print(f"Evaluation completed!")
        print(f"Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
        
    def report_skipped_items(self) -> Dict[str, Any]:
        """
        Generate a report of all skipped items with detailed reasons.
        
        Returns:
            Dict[str, Any]: Dictionary containing skipped items and summary statistics
        """
        if not self.skipped_items:
            return {
                'total_skipped': 0,
                'skipped_by_reason': {},
                'skipped_items': []
            }
        
        # Group by reason
        skipped_by_reason = defaultdict(list)
        for item in self.skipped_items:
            reason = item.get('reason', 'unknown')
            skipped_by_reason[reason].append(item)
        
        # Create summary statistics
        reason_summary = {}
        for reason, items in skipped_by_reason.items():
            reason_summary[reason] = {
                'count': len(items),
                'percentage': len(items) / len(self.skipped_items) * 100
            }
        
        return {
            'total_skipped': len(self.skipped_items),
            'skipped_by_reason': reason_summary,
            'skipped_items': self.skipped_items
        }
        
    def print_skipped_items_summary(self) -> None:
        """
        Print a human-readable summary of skipped items.
        """
        skipped_report = self.report_skipped_items()
        
        if skipped_report['total_skipped'] == 0:
            print("\n📋 SKIPPED ITEMS SUMMARY: No items were skipped!")
            return
        
        print(f"\n📋 SKIPPED ITEMS SUMMARY:")
        print(f"   Total Skipped: {skipped_report['total_skipped']}")
        print(f"\n   Skipped by Reason:")
        
        for reason, stats in skipped_report['skipped_by_reason'].items():
            print(f"     • {reason}: {stats['count']} items ({stats['percentage']:.1f}%)")
        
        print(f"\n   Detailed Skipped Items:")
        for item in skipped_report['skipped_items']:
            id_str = item.get('id', 'unknown')
            line_num = item.get('line_num', 'unknown')
            reason = item.get('reason', 'unknown')
            task_name = item.get('task_name', 'unknown')
            task_type = item.get('task_type', 'unknown')
            
            print(f"     • ID: {id_str} (Line {line_num})")
            print(f"       Reason: {reason}")
            print(f"       Task: {task_name} ({task_type})")
            
            # Add raw answer for parsing failures
            if 'raw_answer' in item:
                raw_answer = item['raw_answer']
                # Truncate long answers
                if len(raw_answer) > 100:
                    raw_answer = raw_answer[:100] + "..."
                print(f"       Raw Answer: {raw_answer}")
            print()
        
    def report_by_task_type(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Generate comprehensive statistics report grouped by task type (question_type and step_num).
        
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Nested dictionary {question_type: {step_num: {metric: value}}}
        """
        if not self.eval_results:
            print("No evaluation results found. Please run evaluate() first.")
            return {}
            
        # Group by question type and step number
        type_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for result in self.eval_results.values():
            task_type = result.get('type', '')
            eval_metrics = result.get('eval_metrics', {})
            
            if not task_type or not eval_metrics:
                continue
                
            # Parse task type (e.g., "forward_dynamics_10_steps")
            parts = task_type.split('_')
            if len(parts) >= 3:
                question_type = '_'.join(parts[:-2])  # e.g., "forward_dynamics"
                step_info = '_'.join(parts[-2:])       # e.g., "10_steps"
                
                # Extract step number
                step_num = ''.join(filter(str.isdigit, step_info))
                if not step_num:
                    step_num = 'unknown'
                    
                # Collect all metrics
                type_groups[question_type][step_num]['match'].append(eval_metrics.get('match', False))
                type_groups[question_type][step_num]['exact_match'].append(eval_metrics.get('exact_match', False))
                type_groups[question_type][step_num]['semantic_match'].append(eval_metrics.get('semantic_match', False))
                type_groups[question_type][step_num]['kendall_tau'].append(eval_metrics.get('kendall_tau', 0.0))
                type_groups[question_type][step_num]['normalized_lcs'].append(eval_metrics.get('normalized_lcs', 0.0))
                
        # Calculate comprehensive statistics for each group
        report = {}
        for question_type, step_groups in type_groups.items():
            report[question_type] = {}
            for step_num, metrics_data in step_groups.items():
                stats = {}
                
                # Calculate accuracy metrics
                match_results = metrics_data['match']
                exact_match_results = metrics_data['exact_match']
                semantic_match_results = metrics_data['semantic_match']
                kendall_tau_values = metrics_data['kendall_tau']
                lcs_values = metrics_data['normalized_lcs']
                
                stats['count'] = len(match_results)
                stats['overall_accuracy'] = sum(match_results) / len(match_results) if match_results else 0.0
                stats['exact_match_accuracy'] = sum(exact_match_results) / len(exact_match_results) if exact_match_results else 0.0
                stats['semantic_match_accuracy'] = sum(semantic_match_results) / len(semantic_match_results) if semantic_match_results else 0.0
                
                # Calculate correlation statistics
                if kendall_tau_values:
                    stats['avg_kendall_tau'] = np.mean(kendall_tau_values)
                    stats['std_kendall_tau'] = np.std(kendall_tau_values)
                    stats['min_kendall_tau'] = np.min(kendall_tau_values)
                    stats['max_kendall_tau'] = np.max(kendall_tau_values)
                
                if lcs_values:
                    stats['avg_normalized_lcs'] = np.mean(lcs_values)
                    stats['std_normalized_lcs'] = np.std(lcs_values)
                    stats['min_normalized_lcs'] = np.min(lcs_values)
                    stats['max_normalized_lcs'] = np.max(lcs_values)
                
                report[question_type][step_num] = stats
                
        return report
        
    def report_by_task_name(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate comprehensive statistics report grouped by task name.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary {task_name: {metric: value}}
        """
        if not self.eval_results:
            print("No evaluation results found. Please run evaluate() first.")
            return {}
            
        # Group by task name
        task_groups = defaultdict(lambda: defaultdict(list))
        
        for result in self.eval_results.values():
            task_name = result.get('task_name')
            eval_metrics = result.get('eval_metrics', {})
            
            if task_name and eval_metrics:
                # Collect all metrics
                task_groups[task_name]['match'].append(eval_metrics.get('match', False))
                task_groups[task_name]['exact_match'].append(eval_metrics.get('exact_match', False))
                task_groups[task_name]['semantic_match'].append(eval_metrics.get('semantic_match', False))
                task_groups[task_name]['kendall_tau'].append(eval_metrics.get('kendall_tau', 0.0))
                task_groups[task_name]['normalized_lcs'].append(eval_metrics.get('normalized_lcs', 0.0))
                
        # Calculate comprehensive statistics for each task
        report = {}
        for task_name, metrics_data in task_groups.items():
            stats = {}
            
            # Calculate accuracy metrics
            match_results = metrics_data['match']
            exact_match_results = metrics_data['exact_match']
            semantic_match_results = metrics_data['semantic_match']
            kendall_tau_values = metrics_data['kendall_tau']
            lcs_values = metrics_data['normalized_lcs']
            
            stats['count'] = len(match_results)
            stats['overall_accuracy'] = sum(match_results) / len(match_results) if match_results else 0.0
            stats['exact_match_accuracy'] = sum(exact_match_results) / len(exact_match_results) if exact_match_results else 0.0
            stats['semantic_match_accuracy'] = sum(semantic_match_results) / len(semantic_match_results) if semantic_match_results else 0.0
            
            # Calculate correlation statistics
            if kendall_tau_values:
                stats['avg_kendall_tau'] = np.mean(kendall_tau_values)
                stats['std_kendall_tau'] = np.std(kendall_tau_values)
                stats['min_kendall_tau'] = np.min(kendall_tau_values)
                stats['max_kendall_tau'] = np.max(kendall_tau_values)
            
            if lcs_values:
                stats['avg_normalized_lcs'] = np.mean(lcs_values)
                stats['std_normalized_lcs'] = np.std(lcs_values)
                stats['min_normalized_lcs'] = np.min(lcs_values)
                stats['max_normalized_lcs'] = np.max(lcs_values)
            
            report[task_name] = stats
            
        return report
        
    def report_overall_score(self) -> Dict[str, Any]:
        """
        Generate comprehensive overall performance report with detailed metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing comprehensive overall metrics
        """
        if not self.eval_results:
            print("No evaluation results found. Please run evaluate() first.")
            return {}
            
        # Collect metrics by dynamics type
        forward_metrics = defaultdict(list)
        inverse_metrics = defaultdict(list)
        all_metrics = defaultdict(list)
        
        for result in self.eval_results.values():
            task_type = result.get('type', '').lower()
            eval_metrics = result.get('eval_metrics', {})
            
            if not eval_metrics:
                continue
            
            # Collect all metrics for overall statistics
            for metric_name in ['match', 'exact_match', 'semantic_match', 'kendall_tau', 'normalized_lcs']:
                metric_value = eval_metrics.get(metric_name, 0.0 if metric_name in ['kendall_tau', 'normalized_lcs'] else False)
                all_metrics[metric_name].append(metric_value)
                
                if 'forward' in task_type:
                    forward_metrics[metric_name].append(metric_value)
                elif 'inverse' in task_type:
                    inverse_metrics[metric_name].append(metric_value)
        
        # Build comprehensive report
        report = {}
        
        # Overall statistics
        if all_metrics['match']:
            report['overall'] = {
                'count': len(all_metrics['match']),
                'overall_accuracy': sum(all_metrics['match']) / len(all_metrics['match']),
                'exact_match_accuracy': sum(all_metrics['exact_match']) / len(all_metrics['exact_match']),
                'semantic_match_accuracy': sum(all_metrics['semantic_match']) / len(all_metrics['semantic_match']),
                'avg_kendall_tau': np.mean(all_metrics['kendall_tau']),
                'std_kendall_tau': np.std(all_metrics['kendall_tau']),
                'min_kendall_tau': np.min(all_metrics['kendall_tau']),
                'max_kendall_tau': np.max(all_metrics['kendall_tau']),
                'avg_normalized_lcs': np.mean(all_metrics['normalized_lcs']),
                'std_normalized_lcs': np.std(all_metrics['normalized_lcs']),
                'min_normalized_lcs': np.min(all_metrics['normalized_lcs']),
                'max_normalized_lcs': np.max(all_metrics['normalized_lcs'])
            }
        
        # Forward dynamics statistics
        if forward_metrics['match']:
            report['forward_dynamics'] = {
                'count': len(forward_metrics['match']),
                'overall_accuracy': sum(forward_metrics['match']) / len(forward_metrics['match']),
                'exact_match_accuracy': sum(forward_metrics['exact_match']) / len(forward_metrics['exact_match']),
                'semantic_match_accuracy': sum(forward_metrics['semantic_match']) / len(forward_metrics['semantic_match']),
                'avg_kendall_tau': np.mean(forward_metrics['kendall_tau']),
                'std_kendall_tau': np.std(forward_metrics['kendall_tau']),
                'min_kendall_tau': np.min(forward_metrics['kendall_tau']),
                'max_kendall_tau': np.max(forward_metrics['kendall_tau']),
                'avg_normalized_lcs': np.mean(forward_metrics['normalized_lcs']),
                'std_normalized_lcs': np.std(forward_metrics['normalized_lcs']),
                'min_normalized_lcs': np.min(forward_metrics['normalized_lcs']),
                'max_normalized_lcs': np.max(forward_metrics['normalized_lcs'])
            }
        
        # Inverse dynamics statistics
        if inverse_metrics['match']:
            report['inverse_dynamics'] = {
                'count': len(inverse_metrics['match']),
                'overall_accuracy': sum(inverse_metrics['match']) / len(inverse_metrics['match']),
                'exact_match_accuracy': sum(inverse_metrics['exact_match']) / len(inverse_metrics['exact_match']),
                'semantic_match_accuracy': sum(inverse_metrics['semantic_match']) / len(inverse_metrics['semantic_match']),
                'avg_kendall_tau': np.mean(inverse_metrics['kendall_tau']),
                'std_kendall_tau': np.std(inverse_metrics['kendall_tau']),
                'min_kendall_tau': np.min(inverse_metrics['kendall_tau']),
                'max_kendall_tau': np.max(inverse_metrics['kendall_tau']),
                'avg_normalized_lcs': np.mean(inverse_metrics['normalized_lcs']),
                'std_normalized_lcs': np.std(inverse_metrics['normalized_lcs']),
                'min_normalized_lcs': np.min(inverse_metrics['normalized_lcs']),
                'max_normalized_lcs': np.max(inverse_metrics['normalized_lcs'])
            }
                
        return report
