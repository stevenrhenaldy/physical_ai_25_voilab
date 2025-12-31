"""
Dataset-based evaluation runner for UMI.
Loads validation data and computes prediction error metrics.

Usage:
    python eval_all_checkpoints.py --exp_dir outputs/train_0 --dataset_path data/combined_1218/dataset.zarr.zip
"""
import os
import sys
import json
import pathlib
import numpy as np
import torch
import zarr
from tqdm import tqdm
import click
import dill
import hydra
import pandas as pd
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.umi_dataset import UmiDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class UmiDatasetEvalRunner:
    def __init__(self, 
                 output_dir,
                 dataset_path,
                 shape_meta,
                 n_eval_episodes=10,
                 max_steps_per_episode=100):
        """
        Args:
            output_dir: Directory to save evaluation results
            dataset_path: Path to the zarr dataset
            shape_meta: Shape metadata from config
            n_eval_episodes: Number of episodes to evaluate
            max_steps_per_episode: Max steps per episode to evaluate
        """
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        self.shape_meta = shape_meta
        self.n_eval_episodes = n_eval_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
    def run(self, policy: BaseImagePolicy):
        """Run evaluation on validation dataset."""
        device = policy.device
        
        # Load dataset
        print(f"Loading dataset from {self.dataset_path}")
        dataset = UmiDataset(
            shape_meta=self.shape_meta,
            dataset_path=self.dataset_path,
            pose_repr={'obs_pose_repr': 'relative', 'action_pose_repr': 'relative'},
            val_ratio=0.05,
            seed=42
        )
        
        # Get validation indices
        val_mask = dataset.val_mask
        val_indices = np.where(val_mask)[0]
        
        if len(val_indices) == 0:
            print("Warning: No validation data found!")
            return {}
        
        print(f"Found {len(val_indices)} validation episodes")
        
        # Evaluate on subset
        n_episodes = min(self.n_eval_episodes, len(val_indices))
        eval_indices = val_indices[:n_episodes]
        
        all_pred_actions = []
        all_gt_actions = []
        all_errors = []
        
        policy.eval()
        
        with torch.no_grad():
            for ep_idx in tqdm(eval_indices, desc="Evaluating"):
                # Get episode data (returns dict with 'obs' dict and 'action' array)
                episode_data = dataset[int(ep_idx)]
                
                # Extract obs and action
                obs_dict = episode_data['obs']
                actions = episode_data['action']
                
                # Get the sequence length from actions
                T = actions.shape[0]
                max_steps = min(self.max_steps_per_episode, T)
                
                # Evaluate each step - need to construct proper obs windows with history
                for step_idx in range(max_steps):
                    # For each obs key, we need to include the history up to step_idx
                    obs = {}
                    for key, value in obs_dict.items():
                        # value has shape (T, ...)
                        # Get the horizon for this key
                        obs_horizon = self.shape_meta['obs'][key].get('horizon', 1)
                        
                        # Get history window: [step_idx - horizon + 1 : step_idx + 1]
                        start_idx = max(0, step_idx - obs_horizon + 1)
                        obs_window = value[start_idx:step_idx + 1]  # Shape: (actual_horizon, ...)
                        
                        # Pad at the beginning if we don't have enough history
                        if obs_window.shape[0] < obs_horizon:
                            pad_len = obs_horizon - obs_window.shape[0]
                            # Create padding with same shape as obs_window but first dim = pad_len
                            pad_shape = list(obs_window.shape)
                            pad_shape[0] = pad_len
                            if isinstance(obs_window, torch.Tensor):
                                padding = torch.zeros(pad_shape, dtype=obs_window.dtype, device=obs_window.device)
                            else:
                                padding = np.zeros(pad_shape, dtype=obs_window.dtype)
                            obs_window = torch.cat([padding, obs_window], dim=0) if isinstance(obs_window, torch.Tensor) else np.concatenate([padding, obs_window], axis=0)
                        
                        obs[key] = obs_window
                    
                    # Add batch dimension and move to device
                    # Data is already torch tensors from dataset
                    obs_batch = dict_apply(obs, lambda x: x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).unsqueeze(0).to(device))
                    
                    # Get prediction
                    result = policy.predict_action(obs_batch)
                    pred_action = result['action'][0].cpu().numpy()  # Remove batch dim
                    
                    # Get ground truth action (convert to numpy if tensor)
                    action_horizon = self.shape_meta['action']['horizon']
                    end_idx = min(step_idx + action_horizon, T)
                    gt_action = actions[step_idx:end_idx]
                    if isinstance(gt_action, torch.Tensor):
                        gt_action = gt_action.cpu().numpy()
                    
                    # Handle dimension mismatch
                    min_len = min(len(pred_action), len(gt_action))
                    pred_action = pred_action[:min_len]
                    gt_action = gt_action[:min_len]
                    
                    all_pred_actions.append(pred_action)
                    all_gt_actions.append(gt_action)
                    
                    # Compute error
                    error = np.abs(pred_action - gt_action)
                    all_errors.append(error)
        
        # Aggregate metrics - pad to same length since horizons vary
        if all_pred_actions:
            max_len = max(len(x) for x in all_pred_actions)
            
            all_pred_actions_padded = []
            all_gt_actions_padded = []
            all_errors_padded = []
            
            for pred, gt, err in zip(all_pred_actions, all_gt_actions, all_errors):
                # Pad with zeros if needed
                if len(pred) < max_len:
                    pad_len = max_len - len(pred)
                    pred = np.pad(pred, ((0, pad_len), (0, 0)), mode='constant')
                    gt = np.pad(gt, ((0, pad_len), (0, 0)), mode='constant')
                    err = np.pad(err, ((0, pad_len), (0, 0)), mode='constant')
                
                all_pred_actions_padded.append(pred)
                all_gt_actions_padded.append(gt)
                all_errors_padded.append(err)
            
            all_pred_actions = np.array(all_pred_actions_padded)
            all_gt_actions = np.array(all_gt_actions_padded)
            all_errors = np.array(all_errors_padded)
        else:
            all_pred_actions = np.array(all_pred_actions)
            all_gt_actions = np.array(all_gt_actions)
            all_errors = np.array(all_errors)
        
        metrics = {
            'n_episodes': n_episodes,
            'n_steps': len(all_errors),
            'mae': float(np.mean(all_errors)),
            'mae_std': float(np.std(all_errors)),
            'mse': float(np.mean(all_errors ** 2)),
            'mae_per_dim': all_errors.mean(axis=(0, 1)).tolist() if all_errors.size > 0 else [],
            'mse_per_dim': (all_errors ** 2).mean(axis=(0, 1)).tolist() if all_errors.size > 0 else [],
        }
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, 'eval_predictions.npz')
        np.savez(
            results_path,
            pred_actions=all_pred_actions,
            gt_actions=all_gt_actions,
            errors=all_errors
        )
        
        print("\n" + "="*50)
        print("Evaluation Results:")
        print("="*50)
        print(f"Evaluated {n_episodes} episodes, {len(all_errors)} steps")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f} ± {metrics['mae_std']:.4f}")
        print(f"Mean Squared Error (MSE):  {metrics['mse']:.4f}")
        print(f"MAE per action dim: {np.array(metrics['mae_per_dim'])}")
        print("="*50)
        
        return metrics


def find_all_checkpoints(exp_dir):
    """Find all checkpoint files in experiment directory."""
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoints_dir):
        if filename.endswith('.ckpt'):
            path = os.path.join(checkpoints_dir, filename)
            
            # Extract epoch number from filename (e.g., epoch=001-step=00500.ckpt)
            if filename == 'latest.ckpt':
                epoch_num = float('inf')
            else:
                try:
                    parts = filename.replace('.ckpt', '').split('-')
                    epoch_num = int(parts[0].split('=')[1])
                except (IndexError, ValueError):
                    epoch_num = float('inf')
            
            checkpoints.append((path, epoch_num, filename))
    
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def load_checkpoint_and_policy(checkpoint_path, output_dir, device='cuda:0'):
    """Load checkpoint and return policy."""
    try:
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        instance = hydra.utils.get_class(cfg._target_)
        workspace = instance(cfg, output_dir=output_dir)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        
        device_obj = torch.device(device)
        policy.to(device_obj)
        policy.eval()
        
        return policy, cfg, workspace
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None


@click.command()
@click.option('--exp_dir', required=True, help='Training experiment output directory')
@click.option('--dataset_path', required=True, help='Path to zarr dataset')
@click.option('--n_eval_episodes', default=20, help='Number of validation episodes to evaluate')
@click.option('--max_steps_per_episode', default=100, help='Max steps per episode')
@click.option('--device', default='cuda:0', help='Device (cuda:0, cpu, etc.)')
@click.option('--skip_latest', is_flag=True, help='Skip latest.ckpt')
def main(exp_dir, dataset_path, n_eval_episodes, max_steps_per_episode, device, skip_latest):
    """Evaluate all checkpoints from a training run."""
    
    exp_dir = os.path.abspath(exp_dir)
    dataset_path = os.path.abspath(dataset_path)
    
    print(f"Experiment: {exp_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Evaluating {n_eval_episodes} episodes\n")
    
    checkpoints = find_all_checkpoints(exp_dir)
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    if skip_latest:
        checkpoints = [c for c in checkpoints if c[2] != 'latest.ckpt']
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    for path, epoch, name in checkpoints:
        print(f"  {name}")
    print()
    
    eval_output_base = os.path.join(exp_dir, 'eval_results')
    pathlib.Path(eval_output_base).mkdir(parents=True, exist_ok=True)
    
    all_metrics = {}
    results_summary = []
    
    for checkpoint_path, epoch, filename in checkpoints:
        epoch_label = f"epoch_{epoch:03d}" if epoch != float('inf') else "latest"
        eval_output_dir = os.path.join(eval_output_base, epoch_label)
        pathlib.Path(eval_output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {filename}")
        print(f"{'='*70}")
        
        policy, cfg, workspace = load_checkpoint_and_policy(checkpoint_path, eval_output_dir, device)
        if policy is None:
            continue
        
        shape_meta = cfg.shape_meta if hasattr(cfg, 'shape_meta') else workspace.shape_meta
        
        try:
            runner = UmiDatasetEvalRunner(
                output_dir=eval_output_dir,
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                n_eval_episodes=n_eval_episodes,
                max_steps_per_episode=max_steps_per_episode
            )
            
            metrics = runner.run(policy)
            all_metrics[epoch_label] = metrics
            
            results_summary.append({
                'checkpoint': filename,
                'epoch': epoch if epoch != float('inf') else 'latest',
                **metrics
            })
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    if results_summary:
        print(f"\n\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}\n")
        
        df = pd.DataFrame(results_summary)
        print(df[['checkpoint', 'mae', 'mae_std', 'mse']].to_string(index=False))
        
        best_idx = df['mae'].idxmin()
        best = df.iloc[best_idx]
        print(f"\nBest: {best['checkpoint']} (MAE: {best['mae']:.4f})")
        
        df.to_csv(os.path.join(eval_output_base, 'summary.csv'), index=False)
        
        with open(os.path.join(eval_output_base, 'metrics.json'), 'w') as f:
            json.dump({k: {k2: v2.tolist() if isinstance(v2, np.ndarray) else v2 
                          for k2, v2 in v.items()} 
                      for k, v in all_metrics.items()}, f, indent=2)
        
        print(f"\nResults saved to: {eval_output_base}/")


if __name__ == '__main__':
    main()
