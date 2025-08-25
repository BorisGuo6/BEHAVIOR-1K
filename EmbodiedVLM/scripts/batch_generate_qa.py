#!/usr/bin/env python3
"""
Batch QA Generation Script

Processes multiple task directories from segmented trajectories and generates
Q&A pairs using the QAGenerationManager class, saving results to JSONL format.

Key parameters:
- --seed: Random seed for reproducible results (default: 42)
- --num-to-sample: Number of sequences to sample for multi-step generators (default: 30)
- --max-qa-num: Maximum number of QA pairs to generate per task (default: 25)
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'OmniGibson'))

try:
    from EmbodiedVLM.heuristics.qa_generation import QAGenerationManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class BatchQAGenerator:
    """
    Processes multiple task directories to generate Q&A pairs using QAGenerationManager.
    """
    
    def __init__(self, input_root: str, output_file: str, raw_data_dir: str, seed: int = 42, num_to_sample: int = 30, max_qa_num: int = 25):
        """
        Initialize the batch QA generator.
        
        Args:
            input_root: Root directory containing segmented task directories
            output_file: Output JSONL file path
            raw_data_dir: Root directory containing raw data
            seed: Random seed for reproducible results
            num_to_sample: Number of sequences to sample for multi-step generators
            max_qa_num: Maximum number of QA pairs to generate per task
        """
        self.input_root = Path(input_root)
        self.output_file = Path(output_file)
        self.raw_data_dir = Path(raw_data_dir)
        self.seed = seed
        self.num_to_sample = num_to_sample
        self.max_qa_num = max_qa_num
        
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root directory not found: {input_root}")
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_task_directories(self):
        """
        Get all valid task directories from the input root.
        
        Returns:
            list: List of task directory paths
        """
        task_dirs = []
        for item in self.input_root.iterdir():
            if item.is_dir():
                # Check if directory contains segmented_scene_graph_0.json
                scene_graph_file = item / "segmented_scene_graph_0.json"
                if scene_graph_file.exists():
                    task_dirs.append(item)
        
        task_dirs.sort()  # Sort for consistent processing order
        return task_dirs
    
    def run(self):
        """
        Run the complete batch QA generation process using QAGenerationManager.
        """
        print("=== BATCH QA GENERATION ===")
        print(f"Input directory: {self.input_root}")
        print(f"Output file: {self.output_file}")
        print()
        
        try:
            # Initialize QA generation manager with the input root
            # The manager will automatically load all valid tasks
            manager = QAGenerationManager(str(self.input_root), str(self.raw_data_dir))
            
            print(f"📚 Loaded {manager.num_tasks} tasks from {self.input_root}")
            
            if manager.num_tasks == 0:
                print("❌ No valid tasks found for QA generation")
                return
            
            # Clear output file at the start
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            if self.output_file.exists():
                self.output_file.unlink()  # Remove existing file
            
            # Define step numbers to generate QAs for
            step_numbers = [3, 4, 5, 6, 7, 8, 9, 10]  # [3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
            
            # Initialize report structure: report[task_name][step_length][qa_type] = count
            qa_report = {}
            
            # Get all task names for report initialization
            task_names = [task_data.task_name for task_data in manager.task_data_list]
            for task_name in task_names:
                qa_report[task_name] = {}
                for step_num in step_numbers:
                    qa_report[task_name][step_num] = {
                        'forward_dynamics': 0,
                        'inverse_dynamics': 0,
                        'total': 0
                    }
            
            # Generate QA pairs for each step number
            for step_num in step_numbers:
                print(f"\n{'='*50}")
                print(f"🔄 Processing Step Number: {step_num}")
                print(f"{'='*50}")
                
                # Generate multi-step forward dynamics Q&A pairs with explicit parameters
                print(f"\n⏭️ Generating Multi-Step Forward Dynamics Q&A pairs (step={step_num}, samples={self.num_to_sample}, max_qa={self.max_qa_num})...")
                forward_stats = manager.generate("multi_forward_dynamics", step_length=step_num, qa_gen_logic="ordering", flush_to_file=str(self.output_file), seed=self.seed, num_to_sample=self.num_to_sample, max_qa_num=self.max_qa_num)
                
                # Generate multi-step inverse dynamics Q&A pairs with explicit parameters
                print(f"\n⏪ Generating Multi-Step Inverse Dynamics Q&A pairs (step={step_num}, samples={self.num_to_sample}, max_qa={self.max_qa_num})...")
                inverse_stats = manager.generate("multi_inverse_dynamics", step_length=step_num, qa_gen_logic="ordering", flush_to_file=str(self.output_file), seed=self.seed, num_to_sample=self.num_to_sample, max_qa_num=self.max_qa_num)
                
                # Update report with actual counts
                for task_name in task_names:
                    forward_count = forward_stats.get(task_name, 0)
                    inverse_count = inverse_stats.get(task_name, 0)
                    
                    qa_report[task_name][step_num]['forward_dynamics'] = forward_count
                    qa_report[task_name][step_num]['inverse_dynamics'] = inverse_count
                    qa_report[task_name][step_num]['total'] = forward_count + inverse_count
                
                # Calculate totals for this step
                step_forward_total = sum(forward_stats.values())
                step_inverse_total = sum(inverse_stats.values())
                step_total = step_forward_total + step_inverse_total
                
                print(f"📊 Step {step_num} Summary: Forward={step_forward_total}, Inverse={step_inverse_total}, Total={step_total}")
            
            # Print final summary and report
            print(f"\n🎉 BATCH QA GENERATION COMPLETE!")
            print("=" * 60)
            print(f"📊 FINAL QA GENERATION REPORT:")
            print("=" * 60)
            
            # Calculate overall totals
            overall_forward = 0
            overall_inverse = 0
            overall_total = 0
            
            # Print summary by step
            for step_num in step_numbers:
                step_forward = sum(qa_report[task_name][step_num]['forward_dynamics'] for task_name in task_names)
                step_inverse = sum(qa_report[task_name][step_num]['inverse_dynamics'] for task_name in task_names)
                step_total = step_forward + step_inverse
                
                overall_forward += step_forward
                overall_inverse += step_inverse
                overall_total += step_total
                
                print(f"Step {step_num:2d}: Forward={step_forward:4d}, Inverse={step_inverse:4d}, Total={step_total:4d}")
            
            print("-" * 60)
            print(f"OVERALL: Forward={overall_forward:4d}, Inverse={overall_inverse:4d}, Total={overall_total:4d}")
            print(f"💾 All QA pairs have been incrementally saved to: {self.output_file}")
            print("🚀 Memory usage optimized by flushing after each task")
            print("=" * 60)
            
            # Print the detailed report dictionary
            print(f"\n📋 Detailed QA Report by Task and Step:")
            print("Format: report[task_name][step_length][qa_type] = count")
            print("-" * 60)
            print(json.dumps(qa_report, indent=2))
            
        except Exception as e:
            print(f"❌ Error during QA generation: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Batch generate QA pairs from segmented trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_root',
        nargs='?',
        default='/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/segmented_replayed_trajecotries',
        help='Root directory containing segmented task directories'
    )
    
    parser.add_argument(
        'raw_data_dir',
        nargs='?',
        default='/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/replayed_trajectories',
        help='Root directory containing raw data'
    )
    
    parser.add_argument(
        'output_file',
        nargs='?',
        default='/home/mll-laptop-1/01_projects/03_behavior_challenge/BehaviorEQA_Dataset/QA/behavior_eqa_ordering.jsonl',
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually generating QA pairs'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--num-to-sample',
        type=int,
        default=25,
        help='Number of sequences to sample for multi-step generators'
    )
    
    parser.add_argument(
        '--max-qa-num',
        type=int,
        default=20,
        help='Maximum number of QA pairs to generate per task'
    )
    
    args = parser.parse_args()
    
    print(f"Input root: {args.input_root}")
    print(f"Raw data dir: {args.raw_data_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Seed: {args.seed}")
    print(f"Num to sample: {args.num_to_sample}")
    print(f"Max QA num: {args.max_qa_num}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No QA pairs will be generated")
        input_root = Path(args.input_root)
        if input_root.exists():
            task_dirs = [d for d in input_root.iterdir() 
                        if d.is_dir() and (d / "segmented_scene_graph_0.json").exists()]
            print(f"Would process {len(task_dirs)} task directories:")
            for task_dir in sorted(task_dirs):
                print(f"  - {task_dir.name}")
        else:
            print(f"Input directory does not exist: {args.input_root}")
        return
    
    try:
        generator = BatchQAGenerator(args.input_root, args.output_file, args.raw_data_dir, seed=args.seed, num_to_sample=args.num_to_sample, max_qa_num=args.max_qa_num)
        print(f"🎯 Using random seed: {args.seed} for reproducible results")
        print(f"📊 Generation parameters: num_to_sample={args.num_to_sample}, max_qa_num={args.max_qa_num}")
        generator.run()
        print("\n🎉 Batch QA generation complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
