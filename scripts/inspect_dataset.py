#!/usr/bin/env python3
"""
Script to inspect a dataset.zarr.zip file and show its structure and contents.

Usage:
    python inspect_dataset.py video/video_1218_1/dataset.zarr.zip
"""

import argparse
import os
import zarr
import numpy as np
import sys

# Add the diffusion_policy package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'packages', 'diffusion_policy', 'src'))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# Register codecs for image compression
register_codecs()


def inspect_dataset(dataset_path):
    """
    Inspect a dataset and print its structure and statistics.
    
    Args:
        dataset_path: Path to dataset.zarr.zip file
    """
    print(f"Inspecting dataset: {dataset_path}")
    print("=" * 80)
    print()
    
    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
    
    print(f"Dataset Statistics:")
    print(f"  Number of episodes: {replay_buffer.n_episodes}")
    print(f"  Total steps: {replay_buffer.n_steps}")
    print()
    
    print(f"Data Keys:")
    for key in replay_buffer.keys():
        data = replay_buffer[key]
        print(f"  {key}:")
        print(f"    Shape: {data.shape}")
        print(f"    Dtype: {data.dtype}")
        if hasattr(data, 'compressor'):
            print(f"    Compressor: {data.compressor}")
        if hasattr(data, 'chunks'):
            print(f"    Chunks: {data.chunks}")
        print()
    
    print(f"Episode Information:")
    episode_ends = replay_buffer.episode_ends
    episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
    
    print(f"  Episode lengths:")
    print(f"    Min: {episode_lengths.min()}")
    print(f"    Max: {episode_lengths.max()}")
    print(f"    Mean: {episode_lengths.mean():.2f}")
    print(f"    Median: {np.median(episode_lengths):.2f}")
    print()
    
    print(f"  First 10 episode lengths: {episode_lengths[:10].tolist()}")
    if len(episode_lengths) > 10:
        print(f"  Last 10 episode lengths: {episode_lengths[-10:].tolist()}")
    print()
    
    print(f"Episode ends: {episode_ends[:10].tolist()}...")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a dataset.zarr.zip file"
    )
    parser.add_argument(
        'dataset_path',
        help='Path to dataset.zarr.zip file'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file does not exist: {args.dataset_path}")
        sys.exit(1)
    
    try:
        inspect_dataset(args.dataset_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
