# Dataset Combination Guide

## Summary

**Yes, you can combine the datasets from `video/video_1218_1` and `video/video_1218_2`!** 

Both datasets have compatible structure with the same data keys and data types, making them suitable for combining into a single training dataset for your diffusion policy model.

## Combined Dataset Results

Successfully combined both datasets into: `data/combined_dataset_1218.zarr.zip`

### Statistics:
- **Total episodes**: 114 (69 from video_1218_1 + 45 from video_1218_2)
- **Total steps**: 87,477
- **Average episode length**: 767 steps
- **Episode length range**: 10 - 1,175 steps

### Data Structure:
Both datasets contain the same 6 keys:
- `camera0_rgb`: RGB images (224x224x3)
- `robot0_demo_end_pose`: End effector target pose (6D)
- `robot0_demo_start_pose`: Start pose (6D) 
- `robot0_eef_pos`: End effector position (3D)
- `robot0_eef_rot_axis_angle`: End effector rotation (axis-angle, 3D)
- `robot0_gripper_width`: Gripper width (1D)

## How to Use the Combined Dataset

### Option 1: Use the Combined Dataset Directly

Simply update your training configuration to point to the combined dataset:

```yaml
# In your config file (e.g., umi_pipeline_configs/*.yaml or training config)
dataset:
  _target_: diffusion_policy.dataset.umi_image_dataset.UmiImageDataset
  dataset_path: data/combined_dataset_1218.zarr.zip
  # ... other parameters
```

### Option 2: Keep Datasets Separate (Not Recommended)

While you could theoretically load both datasets separately and alternate between them during training, this approach is more complex and not natively supported by the existing codebase. The combined approach is simpler and more efficient.

## Scripts Provided

Two utility scripts have been created in the `scripts/` directory:

### 1. `inspect_dataset.py`
Inspect the structure and statistics of a dataset:
```bash
/home/akihito/Steven/voilab/.venv/bin/python scripts/inspect_dataset.py <path_to_dataset.zarr.zip>
```

### 2. `combine_datasets.py`
Combine multiple datasets into one:
```bash
/home/akihito/Steven/voilab/.venv/bin/python scripts/combine_datasets.py \
  --input dataset1.zarr.zip dataset2.zarr.zip dataset3.zarr.zip \
  --output combined_dataset.zarr.zip
```

## Benefits of Combining Datasets

1. **More Training Data**: 114 episodes vs. 69 or 45 individually
2. **Better Generalization**: More diverse demonstrations improve model robustness
3. **Simpler Training**: Single dataset path in configuration
4. **Proper Validation Split**: The framework can properly split a single dataset into train/val
5. **Efficient Loading**: All data loaded once from a single file

## Training Recommendations

When training with the combined dataset:

1. **Adjust validation ratio**: With 114 episodes, consider using a 10-20% validation split:
   ```yaml
   val_ratio: 0.1  # or 0.2
   ```

2. **Consider max_train_episodes**: If you want to limit training data:
   ```yaml
   max_train_episodes: 100  # Use only 100 episodes for training
   ```

3. **Monitor for data imbalance**: If the two original datasets represent different scenarios or difficulty levels, monitor training to ensure the model learns from both equally.

## Next Steps

1. Update your training configuration to use `data/combined_dataset_1218.zarr.zip`
2. Run your training script with the updated config
3. If you collect more data in the future, you can easily combine it with the existing dataset using the `combine_datasets.py` script

## Example Training Command

```bash
# Update your training config to point to the combined dataset, then run:
python train.py --config-name=<your_config> \
  task.dataset.dataset_path=data/combined_dataset_1218.zarr.zip
```
