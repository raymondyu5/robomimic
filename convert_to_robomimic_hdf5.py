#!/usr/bin/env python3
"""
Convert custom point cloud dataset to robomimic-compatible HDF5 format.

This script converts the custom episode-based dataset structure to robomimic's
required HDF5 format, handling point clouds, robot observations, and actions.

Usage:
    python convert_to_robomimic_hdf5.py --input_dir /path/to/data --output_file dataset.hdf5
"""

import os
import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json


class RobomimicHDF5Converter:
    """Converts custom dataset to robomimic HDF5 format."""

    def __init__(
        self,
        input_dir,
        output_file,
        obs_keys=None,
        downsample_points=2048,
        camera_id="CL8384200N1",
        compress_pointclouds=True,
        normalize_actions=True,
        action_range=(-1.0, 1.0),
        include_rgb=False,
        max_episodes=None
    ):
        """
        Initialize the converter.

        Args:
            input_dir: Path to directory containing episode_N folders
            output_file: Output HDF5 file path
            obs_keys: List of observation keys to include from episode data
                     Default: ['joint_positions', 'joint_velocities', 'gripper_position', 'cartesian_position']
            downsample_points: Number of points to downsample point clouds to (None = keep all)
            camera_id: Camera ID pattern to match point cloud files
            compress_pointclouds: Whether to use compression for point clouds
            normalize_actions: Whether to normalize actions to action_range
            action_range: Target range for action normalization (min, max)
            include_rgb: Whether to include RGB images (increases file size significantly)
            max_episodes: Maximum number of episodes to convert (None = all)
        """
        self.input_dir = Path(input_dir)
        self.output_file = output_file
        self.obs_keys = obs_keys or [
            'joint_positions',
            'joint_velocities',
            'gripper_position',
            'cartesian_position'
        ]
        self.downsample_points = downsample_points
        self.camera_id = camera_id
        self.compress_pointclouds = compress_pointclouds
        self.normalize_actions = normalize_actions
        self.action_range = action_range
        self.include_rgb = include_rgb
        self.max_episodes = max_episodes

        # Statistics for normalization
        self.action_stats = {'min': None, 'max': None}

    def find_episodes(self):
        """Find all episode directories in input_dir with valid episode files."""
        all_episode_dirs = sorted([
            d for d in self.input_dir.iterdir()
            if d.is_dir() and d.name.startswith('episode_')
        ], key=lambda x: int(x.name.split('_')[1]))

        # Filter to only include episodes with valid .npy files
        valid_episode_dirs = []
        skipped_episodes = []

        for episode_dir in all_episode_dirs:
            episode_file = episode_dir / f"{episode_dir.name}.npy"
            if episode_file.exists():
                valid_episode_dirs.append(episode_dir)
            else:
                skipped_episodes.append(episode_dir.name)

        if skipped_episodes:
            print(f"Warning: Skipping {len(skipped_episodes)} episodes without .npy files:")
            print(f"  {', '.join(skipped_episodes[:10])}" + (" ..." if len(skipped_episodes) > 10 else ""))

        if self.max_episodes:
            valid_episode_dirs = valid_episode_dirs[:self.max_episodes]

        print(f"Found {len(valid_episode_dirs)} valid episodes (out of {len(all_episode_dirs)} total)")
        return valid_episode_dirs

    def load_episode_data(self, episode_dir):
        """
        Load data for a single episode.

        Returns:
            dict with keys: 'obs_list', 'actions', 'pointclouds', 'rgb_images'
        """
        episode_name = episode_dir.name
        episode_file = episode_dir / f"{episode_name}.npy"

        if not episode_file.exists():
            raise FileNotFoundError(f"Episode file not found: {episode_file}")

        # Load episode data
        episode_data = np.load(episode_file, allow_pickle=True).item()
        obs_list = episode_data['obs']
        actions = episode_data['actions']  # Shape: (T, 22)

        episode_length = len(obs_list)

        # Load point clouds for each timestep
        pointclouds = []
        rgb_images = []

        for t in range(episode_length):
            # Load point cloud
            pcd_file = episode_dir / f"{self.camera_id}_{t:06d}.npy"
            if pcd_file.exists():
                pcd = np.load(pcd_file)  # Shape: (N, 3)

                # Downsample if requested
                if self.downsample_points and pcd.shape[0] > self.downsample_points:
                    indices = np.random.choice(
                        pcd.shape[0],
                        size=self.downsample_points,
                        replace=False
                    )
                    pcd = pcd[indices]

                pointclouds.append(pcd)
            else:
                print(f"Warning: Point cloud not found for {episode_name} timestep {t}")
                # Use empty point cloud as placeholder
                pcd_shape = (self.downsample_points or 0, 3)
                pointclouds.append(np.zeros(pcd_shape, dtype=np.float32))

            # Load RGB image if requested
            if self.include_rgb:
                rgb_file = episode_dir / f"{self.camera_id}_{t:06d}.png"
                if rgb_file.exists():
                    from PIL import Image
                    img = np.array(Image.open(rgb_file))
                    rgb_images.append(img)
                else:
                    rgb_images.append(None)

        return {
            'obs_list': obs_list,
            'actions': actions,
            'pointclouds': pointclouds,
            'rgb_images': rgb_images if self.include_rgb else None
        }

    def compute_action_stats(self, episode_dirs):
        """Compute min/max statistics for action normalization."""
        print("Computing action statistics...")
        all_actions = []

        for episode_dir in tqdm(episode_dirs):
            episode_name = episode_dir.name
            episode_file = episode_dir / f"{episode_name}.npy"

            # Skip if episode file doesn't exist
            if not episode_file.exists():
                print(f"\nWarning: Skipping {episode_name} - episode file not found")
                continue

            episode_data = np.load(episode_file, allow_pickle=True).item()
            all_actions.append(episode_data['actions'])

        if len(all_actions) == 0:
            raise ValueError("No valid episodes found with action data!")

        all_actions = np.concatenate(all_actions, axis=0)
        self.action_stats['min'] = all_actions.min(axis=0)
        self.action_stats['max'] = all_actions.max(axis=0)

        print(f"Action range: [{self.action_stats['min'].min():.4f}, {self.action_stats['max'].max():.4f}]")

    def normalize_action(self, action):
        """Normalize action to target range."""
        if not self.normalize_actions:
            return action

        # Normalize from [min, max] to [target_min, target_max]
        action_min = self.action_stats['min']
        action_max = self.action_stats['max']
        target_min, target_max = self.action_range

        # Avoid division by zero
        action_range = action_max - action_min
        action_range[action_range == 0] = 1.0

        normalized = (action - action_min) / action_range  # [0, 1]
        normalized = normalized * (target_max - target_min) + target_min  # [target_min, target_max]

        return normalized

    def create_hdf5_dataset(self, episode_dirs):
        """Create robomimic-compatible HDF5 file."""

        # Compute action statistics if needed
        if self.normalize_actions:
            self.compute_action_stats(episode_dirs)

        print(f"\nCreating HDF5 file: {self.output_file}")

        with h5py.File(self.output_file, 'w') as f:
            # Create data group
            data_grp = f.create_group('data')

            # Process each episode
            total_steps = 0

            for ep_idx, episode_dir in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
                try:
                    # Load episode data
                    ep_data = self.load_episode_data(episode_dir)
                    obs_list = ep_data['obs_list']
                    actions = ep_data['actions']
                    pointclouds = ep_data['pointclouds']
                    rgb_images = ep_data['rgb_images']

                    episode_length = len(obs_list)
                    total_steps += episode_length

                    # Create demo group
                    demo_grp = data_grp.create_group(f'demo_{ep_idx}')

                    # Add actions (normalize if requested)
                    normalized_actions = self.normalize_action(actions)
                    demo_grp.create_dataset(
                        'actions',
                        data=normalized_actions,
                        dtype=np.float32
                    )

                    # Add rewards and dones (required by robomimic)
                    demo_grp.create_dataset(
                        'rewards',
                        data=np.zeros(episode_length, dtype=np.float32)
                    )
                    demo_grp.create_dataset(
                        'dones',
                        data=np.zeros(episode_length, dtype=bool)
                    )
                    # Mark last step as done
                    demo_grp['dones'][-1] = True

                    # Create obs group
                    obs_grp = demo_grp.create_group('obs')

                    # Add point clouds
                    if pointclouds:
                        pcd_array = np.array(pointclouds, dtype=np.float32)  # (T, N, 3)

                        if self.compress_pointclouds:
                            obs_grp.create_dataset(
                                'pointcloud',
                                data=pcd_array,
                                compression='gzip',
                                compression_opts=4
                            )
                        else:
                            obs_grp.create_dataset(
                                'pointcloud',
                                data=pcd_array,
                                dtype=np.float32
                            )

                    # Add RGB images if included
                    if self.include_rgb and rgb_images:
                        valid_images = [img for img in rgb_images if img is not None]
                        if valid_images:
                            img_array = np.array(valid_images, dtype=np.uint8)  # (T, H, W, 3)
                            obs_grp.create_dataset(
                                'rgb_image',
                                data=img_array,
                                compression='gzip',
                                compression_opts=4
                            )

                    # Add other observations from obs_list
                    for obs_key in self.obs_keys:
                        obs_values = []
                        for obs_dict in obs_list:
                            if obs_key in obs_dict:
                                obs_values.append(obs_dict[obs_key])
                            else:
                                print(f"Warning: {obs_key} not found in {episode_dir.name}")
                                break

                        if len(obs_values) == episode_length:
                            obs_array = np.array(obs_values, dtype=np.float32)  # (T, obs_dim)
                            obs_grp.create_dataset(obs_key, data=obs_array)

                    # Add episode metadata
                    demo_grp.attrs['num_samples'] = episode_length
                    demo_grp.attrs['episode_name'] = episode_dir.name

                except Exception as e:
                    print(f"\nError processing {episode_dir.name}: {e}")
                    continue

            # Add global metadata
            f.attrs['total'] = len(episode_dirs)
            f.attrs['total_steps'] = total_steps
            f.attrs['obs_keys'] = json.dumps(self.obs_keys)
            f.attrs['camera_id'] = self.camera_id

            # env_args is required by robomimic but we don't have a sim environment
            env_meta = {
                "env_name": "RealRobot",
                "type": 1,  # EnvType.GYM_TYPE
                "env_kwargs": {}
            }
            data_grp.attrs['env_args'] = json.dumps(env_meta)

            if self.normalize_actions:
                f.attrs['action_normalized'] = True
                f.attrs['action_min'] = self.action_stats['min']
                f.attrs['action_max'] = self.action_stats['max']
                f.attrs['action_range'] = json.dumps(self.action_range)

            if self.downsample_points:
                f.attrs['downsample_points'] = self.downsample_points

        print(f"\n Conversion complete!")
        print(f"  Total episodes: {len(episode_dirs)}")
        print(f"  Total timesteps: {total_steps}")
        print(f"  Output file: {self.output_file}")
        print(f"  File size: {os.path.getsize(self.output_file) / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom dataset to robomimic HDF5 format"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to directory containing episode folders'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--obs_keys',
        type=str,
        nargs='+',
        default=['joint_positions', 'joint_velocities', 'gripper_position', 'cartesian_position'],
        help='Observation keys to include'
    )
    parser.add_argument(
        '--downsample_points',
        type=int,
        default=2048,
        help='Number of points to downsample point clouds to (0 = no downsampling)'
    )
    parser.add_argument(
        '--camera_id',
        type=str,
        default='CL8384200N1',
        help='Camera ID pattern for point cloud files'
    )
    parser.add_argument(
        '--no_compress',
        action='store_true',
        help='Disable point cloud compression'
    )
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Disable action normalization'
    )
    parser.add_argument(
        '--action_min',
        type=float,
        default=-1.0,
        help='Minimum value for normalized actions'
    )
    parser.add_argument(
        '--action_max',
        type=float,
        default=1.0,
        help='Maximum value for normalized actions'
    )
    parser.add_argument(
        '--include_rgb',
        action='store_true',
        help='Include RGB images (significantly increases file size)'
    )
    parser.add_argument(
        '--max_episodes',
        type=int,
        default=None,
        help='Maximum number of episodes to convert'
    )

    args = parser.parse_args()

    # Create converter
    converter = RobomimicHDF5Converter(
        input_dir=args.input_dir,
        output_file=args.output_file,
        obs_keys=args.obs_keys,
        downsample_points=args.downsample_points if args.downsample_points > 0 else None,
        camera_id=args.camera_id,
        compress_pointclouds=not args.no_compress,
        normalize_actions=not args.no_normalize,
        action_range=(args.action_min, args.action_max),
        include_rgb=args.include_rgb,
        max_episodes=args.max_episodes
    )

    # Find episodes
    episode_dirs = converter.find_episodes()

    if not episode_dirs:
        print(f"Error: No episodes found in {args.input_dir}")
        return

    # Create HDF5 dataset
    converter.create_hdf5_dataset(episode_dirs)


if __name__ == '__main__':
    main()
