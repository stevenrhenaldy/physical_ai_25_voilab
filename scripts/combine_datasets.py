#!/usr/bin/env python3
"""
Script to combine multiple datasets into a single dataset for diffusion policy training.

This script loads multiple dataset.zarr.zip files and combines them into a single
ReplayBuffer by adding all episodes from each dataset sequentially.

Usage:
    python combine_datasets.py --input video/video_1218_1/dataset.zarr.zip video/video_1218_2/dataset.zarr.zip --output data/combined_dataset.zarr.zip
"""

import argparse
import os
import zarr
import numpy as np
from pathlib import Path
import sys

# Add the diffusion_policy package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'packages', 'diffusion_policy', 'src'))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# Register codecs for image compression
register_codecs()


def combine_datasets(input_paths, output_path, verbose=True):
    """
    Combine multiple dataset zarr files into a single combined dataset.
    
    Args:
        input_paths: List of paths to input dataset.zarr.zip files
        output_path: Path to output combined dataset.zarr.zip file
        verbose: Whether to print progress information
    """
    if verbose:
        print(f"Combining {len(input_paths)} datasets...")
        print(f"Input datasets:")
        for path in input_paths:
            print(f"  - {path}")
        print(f"Output: {output_path}")
        print()
    
    # Load the first dataset to initialize the combined buffer
    if verbose:
        print(f"Loading dataset 1/{len(input_paths)}: {input_paths[0]}")
    
    with zarr.ZipStore(input_paths[0], mode='r') as zip_store:
        combined_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
    
    if verbose:
        print(f"  Episodes: {combined_buffer.n_episodes}")
        print(f"  Total steps: {combined_buffer.n_steps}")
        print(f"  Keys: {list(combined_buffer.keys())}")
        print()
    
    # Add episodes from remaining datasets
    for idx, dataset_path in enumerate(input_paths[1:], start=2):
        if verbose:
            print(f"Loading dataset {idx}/{len(input_paths)}: {dataset_path}")
        
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            temp_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, 
                store=zarr.MemoryStore()
            )
        
        if verbose:
            print(f"  Episodes: {temp_buffer.n_episodes}")
            print(f"  Total steps: {temp_buffer.n_steps}")
        
        # Verify that both datasets have the same keys
        if set(combined_buffer.keys()) != set(temp_buffer.keys()):
            raise ValueError(
                f"Dataset keys mismatch!\n"
                f"Combined buffer keys: {set(combined_buffer.keys())}\n"
                f"Dataset {dataset_path} keys: {set(temp_buffer.keys())}"
            )
        
        # Add each episode from the temporary buffer to the combined buffer
        for episode_idx in range(temp_buffer.n_episodes):
            episode_slice = temp_buffer.get_episode(episode_idx)
            combined_buffer.add_episode(episode_slice)
            
            if verbose and (episode_idx + 1) % 10 == 0:
                print(f"  Added {episode_idx + 1}/{temp_buffer.n_episodes} episodes")
        
        if verbose:
            print(f"  Finished adding all {temp_buffer.n_episodes} episodes")
            print()
    
    # Save the combined buffer
    if verbose:
        print(f"Saving combined dataset to {output_path}")
        print(f"  Total episodes: {combined_buffer.n_episodes}")
        print(f"  Total steps: {combined_buffer.n_steps}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the combined buffer
    with zarr.ZipStore(output_path, mode='w') as zip_store:
        combined_buffer.save_to_store(store=zip_store)
    
    if verbose:
        print(f"✓ Successfully saved combined dataset to {output_path}")
        print()
        print("Summary:")
        print(f"  Input datasets: {len(input_paths)}")
        print(f"  Total episodes: {combined_buffer.n_episodes}")
        print(f"  Total steps: {combined_buffer.n_steps}")
        print(f"  Keys: {list(combined_buffer.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple dataset.zarr.zip files into a single dataset"
    )
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        required=True,
        help='Paths to input dataset.zarr.zip files'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output combined dataset.zarr.zip file'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Verify input files exist
    for input_path in args.input:
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist: {input_path}")
            sys.exit(1)
    
    # Check if output file already exists
    if os.path.exists(args.output):
        response = input(f"Output file {args.output} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    try:
        combine_datasets(
            input_paths=args.input,
            output_path=args.output,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
