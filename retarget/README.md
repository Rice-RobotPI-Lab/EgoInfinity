# retarget

Retargets hand motion from egocentric video clips to bilateral robot arm trajectories.

The core challenge is that the camera's position relative to the robot is unknown and varies per clip. A flow-matching neural network (`NeuralRootFrameEstimator`) is trained entirely on synthetic data to estimate the robot's root frame pose from bilateral wrist trajectories observed in the camera frame. IK then solves arm and finger joint angles for the retargeted motion.

Supported robots: **G1** (Unitree), **Franka**, **Robonaut2**, **XLeRobot**.

---

## Installation

```bash
conda env create -f environment.yml
conda activate egoinfinity
```

The environment installs MuJoCo 3.6, MJX (JAX-accelerated sim for GPU training), PyTorch 2.4 (CUDA 12), and pytorch-kinematics for batch IK. GPU is required for training; inference runs on CPU or GPU.

---

## Training

```bash
# G1 with default settings
python3 scripts/train.py

# Different robot
python3 scripts/train.py --robot franka --log_dir runs/franka

# Resume from latest checkpoint
python3 scripts/train.py --robot g1 --log_dir runs/g1

# Resume from a specific checkpoint
python3 scripts/train.py --robot g1 --resume runs/g1/ckpt_epoch_200.pt
```

Training generates synthetic bilateral wrist trajectories on-the-fly using GPU-parallelized MJX simulation (`--num_envs` environments in parallel), projects them into random camera frames, and trains via flow-matching loss.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--robot` | `g1` | Target robot (`g1`, `franka`, `robonaut2`, `xlerobot`) |
| `--num_envs` | `1024` | Parallel MJX environments for data generation |
| `--epochs` | `500` | Number of training epochs |
| `--steps_per_epoch` | `20` | Gradient steps per epoch (each uses a fresh sim batch) |
| `--lr` | `1e-3` | Learning rate |
| `--log_dir` | `runs/{robot}/{mode}` | Checkpoint and TensorBoard log directory |
| `--sample_mode` | `taskspace` | `taskspace`: Cartesian waypoints → IK → FK (more realistic); `jointspace`: random joint angles → FK |
| `--d_model` | `128` | Transformer hidden dimension |
| `--num_layers` | `4` | Number of transformer layers |

### Augmentation options

| Flag | Default | Description |
|------|---------|-------------|
| `--track_pos_noise` | `0.01` | Gaussian noise on wrist positions [m] |
| `--track_ori_noise` | `0.05` | Max orientation noise on wrist poses [rad] |
| `--hand_jump_prob` | `0.2` | Probability of a tracking-jump artifact |
| `--hand_occlusion_prob` | `0.2` | Probability of occluding one arm for a block of frames |
| `--gravity_noise` | `0.1` | Max gravity direction noise [rad] |
| `--gravity_dropout` | `0.3` | Fraction of samples where gravity signal is zeroed |
| `--cam_behind_prob` | `0.15` | Probability of placing the camera in the rear arc |

### TensorBoard

```bash
tensorboard --logdir runs/
```

---

## Testing

### Checkpoints

Preliminary checkpoints (trained with a limited number of steps and randomization) are included under `ckpts/` and available for direct download:

| Robot | File | Download |
|-------|------|----------|
| G1 (Unitree) | `ckpts/g1.pt` | [g1.pt](https://github.com/charlierkj/egoinfinity_retarget/raw/clean/ckpts/g1.pt) |
| Franka FR3 | `ckpts/franka.pt` | [franka.pt](https://github.com/charlierkj/egoinfinity_retarget/raw/clean/ckpts/franka.pt) |
| Robonaut2 | `ckpts/robonaut2.pt` | [robonaut2.pt](https://github.com/charlierkj/egoinfinity_retarget/raw/clean/ckpts/robonaut2.pt) |
| XLeRobot | `ckpts/xlerobot.pt` | [xlerobot.pt](https://github.com/charlierkj/egoinfinity_retarget/raw/clean/ckpts/xlerobot.pt) |

`test.py` resolves `ckpts/<robot>.pt` automatically — no `--ckpt` flag needed if you use the defaults above.

### Retargeting a clip

Retarget a single extracted clip directory to a robot:

```bash
# G1 with the bundled checkpoint
python3 scripts/test.py examples/-QALmP1nHtM_678.2_682.2

# Different robot
python3 scripts/test.py examples/-QALmP1nHtM_678.2_682.2 --robot franka

# Use your own trained checkpoint
python3 scripts/test.py examples/-QALmP1nHtM_678.2_682.2 --robot g1 \
    --ckpt runs/g1/ckpt_epoch_500.pt

# Custom output directory; skip interactive preview
python3 scripts/test.py examples/-QALmP1nHtM_678.2_682.2 --robot g1 \
    --out /results/g1/ --no-preview
```

After saving results, `test.py` opens a MuJoCo viewer to play back the retargeted trajectory at the clip's original FPS. Close the window to exit. Pass `--no-preview` to skip this.

### Clip format

A clip directory must contain:

```
hand_joints.bin   — (T, max_hands, 21, 3) float32  camera-frame MANO keypoints
hand_meta.json    — frame count, hand slots, per-frame handedness
scene.json        — focal length, gravity direction, FPS
depth.mp4         — (optional) video used as background for input_viz.mp4
```

### Outputs

All results are saved to `<clip_parent>/<robot>/` (or `--out`):

| File | Description |
|------|-------------|
| `trajectory.npz` | Arm + finger joint trajectories, joint names, FPS |
| `root_frames.npz` | Per-frame and anchor root frame SE(3) poses |
| `input_viz.mp4` | Wrist trails, root frame axes overlaid on the input video |
| `robot_sim.mp4` | Offscreen MuJoCo render of the retargeted robot |
| `metrics.npz` | IK convergence rates, position/orientation errors, manipulability |

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--robot` | `g1` | Target robot |
| `--ckpt` | `ckpts/<robot>.pt` | Model checkpoint path |
| `--out` | `<clip_parent>/<robot>/` | Output directory |
| `--window_secs` | `2.0` | Sliding window duration for root frame estimation |
| `--window_stride` | `window_len` | Frame stride between windows |
| `--n_clusters` | `5` | K-medoids clusters for anchor selection |
| `--torso_alpha` | `0.3` | Blend between fixed anchor (0) and per-window root translation (1) |
| `--torso_alpha_rot` | `0.7` | Same blend for rotation |
| `--torso_smooth_sigma` | `10.0` | Gaussian smoothing of root trajectory [frames] |
| `--tol_pos` | `0.01` | IK position convergence tolerance [m] |
| `--smooth_sigma` | `0.1` | Joint-space smoothing after IK [sec] |
| `--self_collision` | off | Run gradient-based self-collision post-processing |
| `--no-preview` | off | Skip the interactive MuJoCo viewer after saving |

---

## Other scripts

### `scripts/visualize.py`
Open a MuJoCo viewer showing the robot at a configured pose. Useful for inspecting the model and coordinate axes.

```bash
python3 scripts/visualize.py                   # G1 at start config
python3 scripts/visualize.py --robot franka
python3 scripts/visualize.py --show-body-frames   # overlay torso and wrist axes
```

### `scripts/viz_trajs.py`
Sample random arm trajectories using the same pipeline as training and animate them in a MuJoCo viewer with colored wrist trails. Useful for validating the data distribution before training.

```bash
python3 scripts/viz_trajs.py                   # 10 clips, G1
python3 scripts/viz_trajs.py --robot franka --n_clips 20
python3 scripts/viz_trajs.py --sample_mode jointspace
```

### `scripts/interactive_start_config.py`
Live slider UI for tuning a robot's `start_config` (resting joint angles). Move sliders to find a good neutral pose; press "Print & Quit" to print the config to the terminal.

```bash
python3 scripts/interactive_start_config.py
python3 scripts/interactive_start_config.py --robot franka
```

---

## Module overview

### `models/`

| Module | Description |
|--------|-------------|
| `vn_transformer.py` | `NeuralRootFrameEstimator` — SE(3)-equivariant flow-matching transformer that takes bilateral wrist trajectories in camera frame and outputs the robot root frame pose. Backbone uses Vector Neuron layers for equivariance. |
| `root_opt.py` | `RootPoseOptimizer` — clusters per-window model predictions, scores candidates by IK convergence and manipulability, and returns the best SE(3) anchor. Also runs per-frame batch IK. |
| `collision.py` | `CollisionFilter` — gradient-based post-processor that resolves cross-arm self-collisions using MuJoCo contact detection. |
| `vn_layers.py` | Vector Neuron building blocks (equivariant linear, nonlinearity, pooling). |

### `sim/`

| Module | Description |
|--------|-------------|
| `vec_env_jax.py` | `JaxVecEnv` — JAX/MJX vectorized environment. Runs N robot forward kinematics in parallel on GPU via `jax.vmap + jit`. Used only during training. |
| `traj_sampler.py` | Generates random bilateral arm trajectories (`taskspace_traj`, `random_joint_traj`) and collects wrist-pose outputs from `JaxVecEnv`. |
| `base_env.py` | `BaseEnv` abstract class defining the single-instance environment interface (`reset`, `set_arm_joints`, `set_finger_joints`, `step_joints`, `get_wrist_pose`). |
| `robot_config.py` | `RobotConfig` dataclass — joint groups, end-effector body names, MJCF path. |
| `robots/` | Per-robot `env.py` (single-instance CPU MuJoCo env), `config.py` (viewer/IK/retargeter settings), `sample_config.py` (training trajectory sampling bounds). |

### `kinematics/`

| Module | Description |
|--------|-------------|
| `wrist_ik.py` | `WristIK` — Damped Least Squares IK using pytorch-kinematics. Supports GPU batch solving, joint limit clamping, null-space objectives (manipulability, smoothness, self-collision avoidance). |
| `wilor_retargeter.py` | Retargets WiLOR MANO-21 keypoints to robot finger joint angles. |

### `utils/`

| Module | Description |
|--------|-------------|
| `clip_io.py` | `SamplesSequence` — reads a clip directory (hand keypoints, metadata, scene info). `save_trajectory` — serialises arm + finger trajectories to `.npz`. |
| `pose_utils.py` | Inference helpers: windowed model inference (`estimate_root_poses`), K-medoids anchor selection (`select_best_anchor`), keyframe blending, SLERP interpolation, Gaussian smoothing, camera→root coordinate transform. |
| `viz.py` | Video I/O (`load_video_frames`, `write_video`), offscreen MuJoCo rendering (`render_robot_sim`), OpenCV overlays (coordinate frames, wrist trails, hand keypoints, gravity direction). |
