# Voilab - AI Coding Agent Instructions

## Project Overview

Voilab is a robotics research toolkit combining **Isaac Sim simulation**, **UMI data collection**, and **Diffusion Policy training** for robot manipulation tasks. The project bridges real-world demonstrations with simulated policy learning using a Franka Panda arm.

**Core pipeline**: Real-world UMI data collection → SLAM trajectory processing → Isaac Sim replay with simulated observations → Diffusion Policy training → ROS2-based policy deployment.

## Architecture

### Component Hierarchy

```
voilab (main package)
├── src/voilab/           # Interactive visualization tools
│   ├── applications/     # Jupyter notebook backends
│   ├── utils/           # Data loaders (zarr, replay buffers, ArUco)
│   └── cli.py           # CLI entry points (voilab command)
├── packages/
│   ├── umi/             # UMI SLAM pipeline, trajectory processing
│   └── diffusion_policy/ # Policy training and evaluation
├── scripts/             # Isaac Sim integration scripts
│   ├── launch_isaacsim_workspace.py  # Main simulation launcher
│   ├── generate_data.py              # Data generation in simulation
│   └── registry/        # Task-specific configs (kitchen, dining-room, living-room)
└── nbs/                 # Interactive Voila notebooks (dataset_visualizer, replay_buffer_viewer)
```

### Key Dependencies

- **Isaac Sim v5.1.0**: NVIDIA Omniverse robotics simulator (Docker-based)
- **ROS2 Humble**: Robot communication layer
- **uv**: Package manager for Python (replaces conda)
- **zarr**: Compressed dataset storage format
- **Voila**: Jupyter notebook → web app converter

## Critical Workflows

### 1. Environment Setup

```bash
# ALWAYS use uv for dependency management
make install              # Install dependencies with uv
make install-dev          # Include dev dependencies (JupyterLab, plotting)
make launch-jupyterlab    # Start interactive environment
```

**Never use `pip` or `conda` directly** - this project exclusively uses `uv sync`.

### 2. Data Collection Pipeline

```bash
# Step 1: Process real-world UMI recordings (SLAM + trajectory extraction)
uv run umi run-slam-pipeline umi_pipeline_configs/gopro13_fisheye_2-7k_reconstruct_pipeline_config.yaml --session-dir {data_path}

# Step 2: Replay trajectories in Isaac Sim with simulated observations
uv run voilab launch-simulator --task kitchen --session_dir {data_path}

# Step 3: Train diffusion policy on resulting dataset
uv run packages/diffusion_policy/train.py --config-path=src/diffusion_policy/config --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path={dataset.zarr.zip}
```

**Progress tracking**: Isaac Sim replay creates `.previous_progress.json` in session directories to resume from last completed episode. Delete this file to restart from scratch.

### 3. Docker for Isaac Sim

```bash
# X11 access required for GUI (Linux)
xhost +local:

# Launch Isaac Sim service
docker compose up isaac-sim

# Launch development workspace
docker compose up -d voilab-workspace
docker exec -it <container> bash
```

**Container architecture**: Two services share volumes (`docker-compose.yaml`):
- `isaac-sim`: Full Isaac Sim with ROS2 bridge, GPU access
- `voilab-workspace`: ROS2 development environment without Isaac Sim

## Project-Specific Conventions

### Task Registry Pattern

Task configurations live in `scripts/registry/{task}_registry.py`. Each registry defines:
- `PRELOAD_OBJECTS`: Scene objects to spawn
- `ARUCO_TAG_TRANSLATION/ROTATION`: Coordinate frame reference
- `FRANKA_TRANSLATION/ROTATION`: Robot base pose
- `CAMERA_TRANSLATION/ROTATION`: Camera mount pose
- `is_episode_completed()`: Task success criteria
- `get_config()`: Complete task configuration dict

**Example**: Adding new task requires creating `scripts/registry/new_task_registry.py` and registering in `scripts/registry/__init__.py`.

### ROS2 Integration 3-Layer Architecture

**Strict separation** (see `docs/ros2_integration_design.md`):
1. **Infrastructure Layer**: Generic ROS2 primitives (subscribe/publish/get_data)
2. **Environment Layer**: Manages subscriptions, processes observations, handles robot-specific logic
3. **Runner Layer**: Policy execution lifecycle, episode management

**Critical rule**: Environment layer decides what topics to subscribe to. Infrastructure only provides generic communication - NO business logic.

### Data Format Conventions

- **Datasets**: `.zarr.zip` archives containing `data/` group with time-series arrays
- **Observations**: 
  - `camera_0_rgb`: RGB images (H, W, 3)
  - `robot_eef_pose`: End-effector pose (x, y, z, qw, qx, qy, qz)
  - `robot_eef_rot_axis_angle`: Rotation as axis-angle
  - `gripper_position`: Gripper state (0.0-1.0)
- **Coordinate frames**: 
  - Isaac Sim uses Y-up (positions in meters)
  - Quaternions as (x, y, z, w) in storage, (w, x, y, z) for Isaac Sim API
  - ArUco tag defines task frame origin

### CLI Entry Points

`voilab` CLI defined in `src/voilab/cli.py`:
- `voilab launch-viewer`: Replay buffer viewer
- `voilab launch-dataset-visualizer`: Full dataset review tool
- `voilab launch-simulator`: Isaac Sim with ROS2 (key arguments: `--task`, `--session_dir`, `--scene_only`, `--random_layout`)

### Interactive Development

JupyterLab notebooks in `nbs/` are **runnable as web apps**:
1. Right-click `.ipynb` → "Open with → voila"
2. Backend logic lives in `src/voilab/applications/`
3. URDF files auto-render with built-in 3D viewer (double-click `.urdf` files)

### Makefile Shortcuts

```bash
make launch-workspace        # Docker workspace (cached)
make launch-workspace-force  # Rebuild Docker images
make init-submodule         # Initialize submodules (isaac-ros-ws)
```

## Common Gotchas

1. **Quaternion order**: Storage uses (x,y,z,w), Isaac Sim API requires (w,x,y,z). Use helper `xyzw_to_wxyz()` in registries.
2. **Isaac Sim requires GPU**: Docker containers need nvidia-docker2 runtime, won't work with CPU-only systems.
3. **ROS2 domain**: `ROS_DOMAIN_ID=0` in all containers - change if network conflicts occur.
4. **UV timeouts**: Isaac Sim dependencies are large, use `UV_HTTP_TIMEOUT=600` (set in docker-compose).
5. **ExifTool version**: UMI pipeline requires ExifTool ≥12.5 for GoPro metadata extraction (`make install-exiftool`).

## File Locations

- **Robot models**: `assets/franka_panda/` (URDF + USD variants)
- **Scene assets**: `assets/CADs/*.usd` (3D models for objects)
- **Camera calibration**: `assets/curobo/franka_umi.yaml`, `assets/lula/frank_umi_descriptor.yaml`
- **Pipeline configs**: `umi_pipeline_configs/*.yaml` (GoPro-specific SLAM settings)
- **Documentation**: `docs/` (DOCKER.md, ros2_integration_design.md, data_collection_isaacsim.md)

## Testing & Debugging

- **Dataset inspection**: `uv run scripts/inspect_dataset.py <dataset.zarr.zip>`
- **Combine datasets**: `uv run scripts/combine_datasets.py`
- **Keyboard teleop**: `uv run scripts/keyboard_teleop.py` (for manual control testing)
- **Replay UMI demo**: `uv run scripts/umi_replay.py`

## Integration Points

- **Isaac Sim ↔ ROS2**: Action graphs in `scripts/action_graph.py` define simulation pipeline nodes (clock, joint states, camera publishers)
- **UMI ↔ Diffusion Policy**: Dataset format compatibility ensured by `packages/umi` preprocessing
- **Voila ↔ JupyterLab**: `jupyterlab-urdf` extension enables URDF visualization

When modifying Isaac Sim scenes, update corresponding registry configs. When adding new observation types, update both environment layer ROS2 subscriptions AND dataset loaders in `src/voilab/utils/`.
