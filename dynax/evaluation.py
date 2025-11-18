"""Optimized evaluation utilities for dynamics models."""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mujoco import mjx

matplotlib.use("Agg")  # Non-interactive backend

from dynax.base_neural_model import BaseNeuralModel, NeuralModelParams
from dynax.envs import Env
from dynax.utils.data import DynamicsDataset, extract_state_features


def render_trajectory_video(
    env: Env,
    true_states: jax.Array,
    pred_states: jax.Array,
    fps: int = 30,
    save_path: Optional[Path] = None,
    max_frames: int = 200,
) -> np.ndarray:
    """Render video showing both true and predicted trajectories.

    The true trajectory is rendered normally, and the predicted trajectory
    is rendered with transparency and light blue color.

    Args:
        env: Environment with model and renderer.
        true_states: True state trajectory, shape (T+1, state_dim).
        pred_states: Predicted state trajectory, shape (T+1, state_dim).
        fps: Frames per second for the video.
        save_path: Optional path to save video file.
        max_frames: Maximum number of frames to render (for speed).

    Returns:
        Video frames with shape (T, H, W, C) for moviepy.
    """
    import subprocess
    import mujoco
    from mujoco import mjx

    sim_dt = env.dt
    render_dt = 1.0 / fps
    render_every = int(round(render_dt / sim_dt))
    num_steps = len(true_states)
    steps = np.arange(0, num_steps, render_every)

    # Limit number of frames for speed
    if len(steps) > max_frames:
        step_indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
        steps = steps[step_indices]

    nq = env.model.nq

    # Convert all states to numpy once (faster)
    true_states_np = np.array(true_states)
    pred_states_np = np.array(pred_states)

    # Store original colors for restoration
    original_rgba = env.mj_model.geom_rgba.copy()

    # Create a second renderer for predicted trajectory (same resolution)
    pred_renderer = mujoco.Renderer(env.mj_model, width=1280, height=720)
    # Enable lighting for predicted renderer too
    pred_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    pred_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
    pred_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False

    # Predicted trajectory color (light blue)
    pred_color = np.array([0.5, 0.8, 1.0, 0.7])  # RGBA: 70% opacity

    # Use FFmpeg for faster video encoding if save_path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Get renderer dimensions
        render_width = env.renderer.width
        render_height = env.renderer.height

        # Set up FFmpeg process for direct streaming
        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{render_width}x{render_height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(fps),
                "-i",
                "-",  # Input from stdin
                "-an",  # No audio
                "-vcodec",
                "h264",
                "-crf",
                "23",  # Good quality, faster than crf=1
                "-preset",
                "fast",  # Faster encoding
                "-pix_fmt",
                "yuv420p",
                "-loglevel",
                "error",
                str(save_path),
            ]

            ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            # Render frames and stream directly to FFmpeg
            for i in steps:
                # Use pre-converted numpy arrays
                true_state = true_states_np[i]
                pred_state = pred_states_np[i]

                # True trajectory: render with normal colors (default model)
                true_data = mjx.make_data(env.model)
                true_data = true_data.replace(
                    qpos=jnp.array(true_state[:nq]),
                    qvel=jnp.array(true_state[nq:]),
                )
                true_data = mjx.forward(env.model, true_data)
                true_mj_data = mjx.get_data(env.mj_model, true_data)

                # Render true trajectory first (normal colors, with lighting)
                env.renderer.update_scene(true_mj_data)
                true_frame = env.renderer.render()  # H, W, C (uint8)

                # Temporarily modify colors for predicted trajectory
                env.mj_model.geom_rgba[:] = pred_color
                # Predicted trajectory: render with modified colors
                pred_data = mjx.make_data(env.model)
                pred_data = pred_data.replace(
                    qpos=jnp.array(pred_state[:nq]),
                    qvel=jnp.array(pred_state[nq:]),
                )
                pred_data = mjx.forward(env.model, pred_data)
                pred_mj_data = mjx.get_data(env.mj_model, pred_data)

                # Render predicted trajectory (colored, with lighting)
                pred_renderer.update_scene(pred_mj_data)
                pred_frame = pred_renderer.render()  # H, W, C (uint8)

                # Restore original colors immediately
                env.mj_model.geom_rgba[:] = original_rgba

                # Create mask for predicted geometry (differs from background)
                bg_color = np.array([135.0, 206.0, 250.0])  # Sky color
                diff_from_bg = np.linalg.norm(
                    pred_frame.astype(np.float32)
                    - bg_color[None, None, :],
                    axis=2,
                )
                pred_mask = (diff_from_bg > 30).astype(np.float32)[:, :, None]

                # Tint predicted frame to light blue
                pred_tinted = pred_frame.astype(np.float32).copy()
                # Reduce red, slight green, increase blue for light blue tint
                pred_tinted[:, :, 0] = np.clip(
                    pred_tinted[:, :, 0] * 0.5, 0, 255
                )
                pred_tinted[:, :, 1] = np.clip(
                    pred_tinted[:, :, 1] * 0.8, 0, 255
                )
                pred_tinted[:, :, 2] = np.clip(
                    pred_tinted[:, :, 2] * 1.2, 0, 255
                )

                # Alpha composite: blend predicted over true
                alpha = pred_color[3]  # 0.7 = 70% opacity
                blended = (
                    pred_tinted * alpha * pred_mask
                    + true_frame.astype(np.float32) * (1 - alpha * pred_mask)
                ).astype(np.uint8)

                # Write frame directly to FFmpeg (RGB format)
                ffmpeg_process.stdin.write(blended.tobytes())

            # Close FFmpeg process
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()

            # Return empty array since we streamed directly
            return np.array([])

        except (subprocess.SubprocessError, FileNotFoundError):
            # Fallback to moviepy if FFmpeg not available
            pass

    # Fallback: collect frames and use moviepy (slower)
    frames = []
    for i in steps:
        # Use pre-converted numpy arrays
        true_state = true_states_np[i]
        pred_state = pred_states_np[i]

        # True trajectory: render with normal colors (default model)
        true_data = mjx.make_data(env.model)
        true_data = true_data.replace(
            qpos=jnp.array(true_state[:nq]), qvel=jnp.array(true_state[nq:])
        )
        true_data = mjx.forward(env.model, true_data)
        true_mj_data = mjx.get_data(env.mj_model, true_data)

        # Render true trajectory first (normal colors, with lighting)
        env.renderer.update_scene(true_mj_data)
        true_frame = env.renderer.render().astype(np.float32)  # H, W, C

        # Temporarily modify colors for predicted trajectory
        env.mj_model.geom_rgba[:] = pred_color
        # Predicted trajectory: render with modified colors
        pred_data = mjx.make_data(env.model)
        pred_data = pred_data.replace(
            qpos=jnp.array(pred_state[:nq]), qvel=jnp.array(pred_state[nq:])
        )
        pred_data = mjx.forward(env.model, pred_data)
        pred_mj_data = mjx.get_data(env.mj_model, pred_data)

        # Render predicted trajectory (colored, with lighting)
        pred_renderer.update_scene(pred_mj_data)
        pred_frame = pred_renderer.render().astype(np.float32)  # H, W, C

        # Restore original colors immediately
        env.mj_model.geom_rgba[:] = original_rgba

        # Create mask for predicted geometry (differs from background)
        bg_color = np.array([135.0, 206.0, 250.0])  # Sky color
        diff_from_bg = np.linalg.norm(
            pred_frame - bg_color[None, None, :], axis=2
        )
        pred_mask = (diff_from_bg > 30).astype(np.float32)[:, :, None]

        # Tint predicted frame to light blue
        pred_tinted = pred_frame.copy()
        # Reduce red, slight green, increase blue for light blue tint
        pred_tinted[:, :, 0] = np.clip(pred_tinted[:, :, 0] * 0.5, 0, 255)
        pred_tinted[:, :, 1] = np.clip(pred_tinted[:, :, 1] * 0.8, 0, 255)
        pred_tinted[:, :, 2] = np.clip(pred_tinted[:, :, 2] * 1.2, 0, 255)

        # Alpha composite: blend predicted over true
        alpha = pred_color[3]  # 0.7 = 70% opacity
        blended = (
            pred_tinted * alpha * pred_mask
            + true_frame * (1 - alpha * pred_mask)
        ).astype(np.uint8)

        frames.append(blended)

    frames = np.stack(frames)  # Shape: (T, H, W, C)

    # Save video if path provided (fallback to moviepy)
    if save_path is not None:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert frames to list for moviepy (expects list of HWC arrays)
        frame_list = [frame for frame in frames]

        # Create video clip with high quality settings
        clip = ImageSequenceClip(frame_list, fps=fps)
        clip.write_videofile(
            str(save_path),
            fps=fps,
            codec="libx264",
            bitrate="8000k",  # High bitrate for quality
            preset="medium",  # Balance between speed and compression
            logger=None,
        )

    return frames


def render_single_trajectory_video(
    env: Env,
    states: jax.Array,
    fps: int = 30,
    save_path: Optional[Path] = None,
    max_frames: int = 500,
) -> None:
    """Render a simple video of a single trajectory (optimized for speed).

    Uses CPU MuJoCo directly for faster rendering.

    Args:
        env: Environment with model and renderer.
        states: State trajectory, shape (T+1, state_dim).
        fps: Frames per second for the video.
        save_path: Path to save video file.
        max_frames: Maximum number of frames to render.
    """
    import subprocess
    import mujoco

    sim_dt = env.dt
    render_dt = 1.0 / fps
    render_every = int(round(render_dt / sim_dt))
    num_steps = len(states)
    steps = np.arange(0, num_steps, render_every)

    # Limit number of frames for speed
    if len(steps) > max_frames:
        step_indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
        steps = steps[step_indices]

    nq = env.mj_model.nq
    states_np = np.array(states)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Get renderer dimensions
    render_width = env.renderer.width
    render_height = env.renderer.height

    # Create CPU MuJoCo data object (faster than MJX for single-threaded)
    mj_data = mujoco.MjData(env.mj_model)

    # Set up FFmpeg process for direct streaming
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{render_width}x{render_height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "h264",
            "-crf",
            "23",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-loglevel",
            "error",
            str(save_path),
        ]

        ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        # Render frames and stream directly to FFmpeg
        for i in steps:
            state = states_np[i]

            # Set state directly in CPU MuJoCo data
            mj_data.qpos[:] = state[:nq]
            mj_data.qvel[:] = state[nq:]

            # Forward kinematics (CPU MuJoCo is faster for single operations)
            mujoco.mj_forward(env.mj_model, mj_data)

            # Render frame
            env.renderer.update_scene(mj_data)
            frame = env.renderer.render()  # H, W, C (uint8)

            # Write frame directly to FFmpeg (RGB format)
            ffmpeg_process.stdin.write(frame.tobytes())

        # Close FFmpeg process
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    except (subprocess.SubprocessError, FileNotFoundError):
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg for video rendering."
        )


def evaluate_single_step(
    model: BaseNeuralModel,
    params: NeuralModelParams,
    states: jax.Array,
    actions: jax.Array,
    next_states_true: jax.Array,
) -> Dict[str, jax.Array]:
    """Evaluate single-step prediction accuracy.

    Args:
        model: Dynamics model.
        params: Model parameters.
        states: Current states, shape (N, state_dim).
        actions: Actions, shape (N, action_dim).
        next_states_true: True next states, shape (N, state_dim).

    Returns:
        Dictionary with metrics:
        - mae: Mean absolute error per dimension, shape (state_dim,).
        - rmse: Root mean squared error per dimension, shape (state_dim,).
        - mae_mean: Overall mean absolute error (scalar).
        - rmse_mean: Overall root mean squared error (scalar).
        - errors: Absolute errors, shape (N, state_dim).
    """
    # Vectorized prediction
    next_states_pred = jax.vmap(
        lambda s, a: model.step(params, s, a)
    )(states, actions)

    # Compute errors
    errors = jnp.abs(next_states_pred - next_states_true)
    squared_errors = jnp.square(next_states_pred - next_states_true)

    # Per-dimension metrics
    mae = jnp.mean(errors, axis=0)
    rmse = jnp.sqrt(jnp.mean(squared_errors, axis=0))

    # Overall metrics
    mae_mean = jnp.mean(errors)
    rmse_mean = jnp.sqrt(jnp.mean(squared_errors))

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_mean": mae_mean,
        "rmse_mean": rmse_mean,
        "errors": errors,
        "next_states_pred": next_states_pred,
        "next_states_true": next_states_true,
    }


def evaluate_single_step_dataset(
    model: BaseNeuralModel,
    params: NeuralModelParams,
    dataset: DynamicsDataset,
    num_samples: Optional[int] = None,
    rng: Optional[jax.Array] = None,
) -> Dict[str, jax.Array]:
    """Evaluate single-step prediction on a dataset.

    Args:
        model: Dynamics model.
        params: Model parameters.
        dataset: Dataset to evaluate on.
        num_samples: Number of samples to evaluate (None = all).
        rng: Random number generator key for sampling.

    Returns:
        Dictionary with metrics (see evaluate_single_step).
    """
    if num_samples is None or num_samples >= len(dataset):
        states = dataset.states
        actions = dataset.actions
        next_states_true = dataset.next_states
    else:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        indices = jax.random.choice(
            rng, len(dataset), (num_samples,), replace=False
        )
        states = dataset.states[indices]
        actions = dataset.actions[indices]
        next_states_true = dataset.next_states[indices]

    return evaluate_single_step(model, params, states, actions, next_states_true)


def create_rollout_fn(
    model: BaseNeuralModel,
    params: NeuralModelParams,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Create a JIT-compiled rollout function with history buffer support.

    Args:
        model: Dynamics model.
        params: Model parameters.

    Returns:
        JIT-compiled function that takes (initial_states, actions) and returns
        trajectory of states. Initial_states has shape (history_length, state_dim).
    """

    @jax.jit
    def rollout_fn(
        initial_states: jax.Array, actions: jax.Array
    ) -> jax.Array:
        """Roll out trajectory using neural dynamics model with history buffer.

        Args:
            initial_states: Initial state history, shape (history_length, state_dim).
            actions: Actions sequence, shape (T, action_dim).

        Returns:
            State trajectory, shape (T+1, state_dim).
            First state is the most recent from initial_states.
        """
        history_length = model.history_length

        def step_fn(
            carry: Tuple[jax.Array, jax.Array], action: jax.Array
        ) -> Tuple:
            """Step with history buffer."""
            state_history, action_history = carry

            # Update action history
            action_history = jnp.concatenate(
                [action_history[1:], action[None, :]], axis=0
            )

            # Predict next state
            next_state = model.step(params, state_history, action_history)

            # Update state history
            state_history = jnp.concatenate(
                [state_history[1:], next_state[None, :]], axis=0
            )

            return (state_history, action_history), next_state

        # Initialize action history with zeros
        action_history = jnp.zeros((history_length, actions.shape[-1]))

        # Run scan
        _, states = jax.lax.scan(
            step_fn, (initial_states, action_history), actions
        )

        # Prepend the most recent initial state
        return jnp.concatenate([initial_states[-1:], states], axis=0)

    return rollout_fn


def create_true_rollout_fn(
    env: Env,
) -> Callable[[mjx.Data, jax.Array], jax.Array]:
    """Create a JIT-compiled true dynamics rollout function.

    Args:
        env: Environment with MuJoCo model.

    Returns:
        JIT-compiled function that takes (initial_data, actions) and returns
        trajectory of state features.
    """

    @jax.jit
    def rollout_fn(initial_data: mjx.Data, actions: jax.Array) -> jax.Array:
        """Roll out trajectory using true MuJoCo dynamics.

        Args:
            initial_data: Initial MJX data.
            actions: Actions sequence, shape (T, action_dim).

        Returns:
            State trajectory, shape (T+1, state_dim).
        """

        def step_fn(data, action):
            data = data.replace(ctrl=action)
            next_data = mjx.step(env.model, data)
            return next_data, extract_state_features(next_data)

        _, states = jax.lax.scan(step_fn, initial_data, actions)
        initial_features = extract_state_features(initial_data)
        return jnp.concatenate([initial_features[None, :], states], axis=0)

    return rollout_fn


def evaluate_rollout(
    neural_rollout_fn: Callable,
    true_rollout_fn: Callable,
    initial_states: jax.Array,
    initial_data: mjx.Data,
    actions: jax.Array,
) -> Dict[str, jax.Array]:
    """Evaluate multi-step rollout accuracy.

    Args:
        neural_rollout_fn: Neural model rollout function.
        true_rollout_fn: True dynamics rollout function.
        initial_states: Initial state history, shape (history_length, state_dim).
        initial_data: Initial MJX data.
        actions: Actions sequence, shape (T, action_dim).

    Returns:
        Dictionary with metrics:
        - true_states: True state trajectory, shape (T+1, state_dim).
        - pred_states: Predicted state trajectory, shape (T+1, state_dim).
        - errors: Absolute errors, shape (T+1, state_dim).
        - mae_over_time: Mean absolute error over time, shape (T+1,).
        - rmse_over_time: Root mean squared error over time, shape (T+1,).
        - final_mae: Final step MAE (scalar).
        - final_rmse: Final step RMSE (scalar).
    """
    # Run rollouts
    true_states = true_rollout_fn(initial_data, actions)
    pred_states = neural_rollout_fn(initial_states, actions)

    # Compute errors
    errors = jnp.abs(pred_states - true_states)
    squared_errors = jnp.square(pred_states - true_states)

    # Metrics over time
    mae_over_time = jnp.mean(errors, axis=1)
    rmse_over_time = jnp.sqrt(jnp.mean(squared_errors, axis=1))

    # Final step metrics
    final_mae = mae_over_time[-1]
    final_rmse = rmse_over_time[-1]

    return {
        "true_states": true_states,
        "pred_states": pred_states,
        "errors": errors,
        "mae_over_time": mae_over_time,
        "rmse_over_time": rmse_over_time,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
    }


def evaluate_rollouts_batch(
    model: mjx.Model,
    neural_rollout_fn: Callable,
    initial_states: jax.Array,
    actions_batch: jax.Array,
    return_trajectories: bool = False,
) -> Dict[str, jax.Array]:
    """Evaluate multiple rollouts in parallel using vmap.

    Args:
        model: MJX model for true dynamics.
        neural_rollout_fn: Neural model rollout function.
        initial_states: Initial state histories, shape (N, history_length, state_dim).
        actions_batch: Actions sequences, shape (N, T, action_dim).
        return_trajectories: If True, also return individual trajectories.

    Returns:
        Dictionary with metrics averaged over rollouts:
        - mae_over_time: Mean absolute error over time, shape (T+1,).
        - rmse_over_time: Root mean squared error over time, shape (T+1,).
        - final_mae: Final step MAE (scalar).
        - final_rmse: Final step RMSE (scalar).
        - true_trajectories: (if return_trajectories=True) True trajectories, shape (N, T+1, state_dim).
        - pred_trajectories: (if return_trajectories=True) Predicted trajectories, shape (N, T+1, state_dim).
    """

    def single_rollout_eval(initial_state_history, actions):
        """Evaluate single rollout (vmappable)."""
        # Create initial MJX data from most recent state in history
        # initial_state_history has shape (history_length, state_dim)
        initial_state = initial_state_history[-1, :]  # Most recent state
        initial_data = mjx.make_data(model)
        nq = model.nq
        initial_data = initial_data.replace(
            qpos=initial_state[:nq], qvel=initial_state[nq:]
        )
        initial_data = mjx.forward(model, initial_data)

        # True rollout
        def true_step(data, action):
            data = data.replace(ctrl=action)
            next_data = mjx.step(model, data)
            state_features = jnp.concatenate([next_data.qpos, next_data.qvel])
            return next_data, state_features

        _, true_states = jax.lax.scan(true_step, initial_data, actions)
        initial_features = jnp.concatenate([initial_data.qpos, initial_data.qvel])
        true_states = jnp.concatenate([initial_features[None, :], true_states], axis=0)

        # Neural rollout
        pred_states = neural_rollout_fn(initial_state_history, actions)

        # Compute errors
        errors = jnp.abs(pred_states - true_states)
        squared_errors = jnp.square(pred_states - true_states)

        mae_over_time = jnp.mean(errors, axis=1)
        rmse_over_time = jnp.sqrt(jnp.mean(squared_errors, axis=1))

        result = {
            "mae_over_time": mae_over_time,
            "rmse_over_time": rmse_over_time,
            "final_mae": mae_over_time[-1],
            "final_rmse": rmse_over_time[-1],
        }

        if return_trajectories:
            result["true_states"] = true_states
            result["pred_states"] = pred_states

        return result

    # Vmap over all rollouts in parallel
    results = jax.vmap(single_rollout_eval)(initial_states, actions_batch)

    # Average metrics
    output = {
        "mae_over_time": jnp.mean(results["mae_over_time"], axis=0),
        "rmse_over_time": jnp.mean(results["rmse_over_time"], axis=0),
        "final_mae": jnp.mean(results["final_mae"]),
        "final_rmse": jnp.mean(results["final_rmse"]),
    }

    if return_trajectories:
        output["true_trajectories"] = results["true_states"]
        output["pred_trajectories"] = results["pred_states"]

    return output


def plot_trajectories(
    true_trajectories: jax.Array,
    pred_trajectories: jax.Array,
    state_dim: int,
    nq: int,
    num_plots: int = 5,
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """Create trajectory comparison plots.

    Args:
        true_trajectories: True trajectories, shape (N, T+1, state_dim).
        pred_trajectories: Predicted trajectories, shape (N, T+1, state_dim).
        state_dim: Total state dimension.
        nq: Number of position dimensions.
        num_plots: Number of rollouts to plot.
        save_path: Optional path to save the figure.

    Returns:
        Figure as numpy array for TensorBoard (shape: H, W, 3).
    """
    num_rollouts = min(num_plots, len(true_trajectories))

    # Create subplots: one row per rollout, one column per state dimension
    # Use smaller figure size and lower DPI for faster rendering
    fig, axes = plt.subplots(
        num_rollouts, state_dim, figsize=(2.5 * state_dim, 1.5 * num_rollouts), dpi=100
    )
    if num_rollouts == 1:
        axes = axes[None, :]
    if state_dim == 1:
        axes = axes[:, None]

    time_steps = np.arange(len(true_trajectories[0]))

    # Convert trajectories to numpy once (faster)
    true_trajs_np = np.array(true_trajectories[:num_rollouts])
    pred_trajs_np = np.array(pred_trajectories[:num_rollouts])

    for rollout_idx in range(num_rollouts):
        true_traj = true_trajs_np[rollout_idx]
        pred_traj = pred_trajs_np[rollout_idx]

        for dim_idx in range(state_dim):
            ax = axes[rollout_idx, dim_idx]

            # Use thinner lines and simpler styling for speed
            ax.plot(time_steps, true_traj[:, dim_idx], "b-", label="True", linewidth=1.5)
            ax.plot(
                time_steps, pred_traj[:, dim_idx], "r--", label="Pred", linewidth=1.5
            )

            # Label first row
            if rollout_idx == 0:
                if dim_idx < nq:
                    ax.set_title(f"Pos {dim_idx}", fontsize=9)
                else:
                    ax.set_title(f"Vel {dim_idx - nq}", fontsize=9)

            # Label first column
            if dim_idx == 0:
                ax.set_ylabel(f"R{rollout_idx + 1}", fontsize=8)

            ax.grid(True, alpha=0.2, linewidth=0.5)
            if rollout_idx == 0 and dim_idx == 0:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(pad=0.5)

    # Save figure if path provided (lower DPI for speed)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    # Convert to numpy array for TensorBoard (faster method)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    buf = buf.reshape((height, width, 4))[:, :, :3]  # Remove alpha channel

    plt.close(fig)

    return buf

