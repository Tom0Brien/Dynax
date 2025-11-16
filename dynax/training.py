"""Training utilities for neural network dynamics models."""

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass
from tensorboardX import SummaryWriter

from dynax.base import BaseDynamicsModel, DynamicsModelParams
from dynax.utils.data import DynamicsDataset
from dynax.utils.normalization import (
    compute_action_normalization_stats,
    compute_normalization_stats,
    denormalize_output,
    normalize_action,
    normalize_state,
)


@dataclass
class TrainingConfig:
    """Configuration for dynamics model training.

    Attributes:
        learning_rate: Learning rate for optimizer (paper: 0.001).
        batch_size: Batch size for training (paper: 512).
        num_epochs: Number of training epochs.
        weight_decay: L2 regularization weight.
        grad_clip: Maximum gradient norm (None for no clipping).
        noise_std: Std dev of Gaussian noise for input/output augmentation.
        loss_fn: Optional custom loss function with signature
            (params, states, actions, targets) -> (loss, metrics).
        log_dir: Optional directory for TensorBoard logging. If None, no logging.
        render_videos: Whether to render videos (slow, disabled by default).
    """

    learning_rate: float = 1e-3
    batch_size: int = 512
    num_epochs: int = 100
    weight_decay: float = 1e-4
    grad_clip: float = None
    noise_std: float = 0.01
    loss_fn: Callable = None
    log_dir: Optional[str | Path] = None
    render_videos: bool = False


@dataclass
class TrainingState:
    """State of the training process.

    Attributes:
        params: Model parameters.
        opt_state: Optimizer state.
        epoch: Current epoch number.
        best_val_loss: Best validation loss seen so far.
    """

    params: dict
    opt_state: optax.OptState
    epoch: int
    best_val_loss: float


def create_loss_fn(
    model: BaseDynamicsModel,
    state_mean: jax.Array,
    state_std: jax.Array,
    action_mean: jax.Array,
    action_std: jax.Array,
    output_mean: jax.Array,
    output_std: jax.Array,
    weight_decay: float = 0.0,
    noise_std: float = 0.0,
) -> Callable:
    """Create a loss function for training the dynamics model.

    Args:
        model: Dynamics model.
        state_mean: State normalization mean.
        state_std: State normalization std.
        action_mean: Action normalization mean.
        action_std: Action normalization std.
        output_mean: Output normalization mean.
        output_std: Output normalization std.
        weight_decay: L2 regularization weight.
        noise_std: Gaussian noise std for data augmentation.
    """

    def loss_fn(
        params: dict,
        states: jax.Array,
        actions: jax.Array,
        targets: jax.Array,
        rng: jax.Array = None,
    ) -> Tuple[jax.Array, dict]:
        """Compute MSE loss between predicted and actual outputs."""
        # Add Gaussian noise for robustness (if noise_std > 0)
        if noise_std > 0 and rng is not None:
            rng_state, rng_action, rng_target = jax.random.split(rng, 3)
            states = states + jax.random.normal(rng_state, states.shape) * noise_std
            actions = actions + jax.random.normal(rng_action, actions.shape) * noise_std
            targets = targets + jax.random.normal(rng_target, targets.shape) * noise_std

        # Normalize states, actions, and targets
        states_norm = jax.vmap(normalize_state, in_axes=(0, None, None))(
            states, state_mean, state_std
        )
        actions_norm = jax.vmap(normalize_action, in_axes=(0, None, None))(
            actions, action_mean, action_std
        )
        targets_norm = jax.vmap(normalize_state, in_axes=(0, None, None))(
            targets, output_mean, output_std
        )

        # Predict normalized outputs
        pred_outputs_norm = jax.vmap(model.apply, in_axes=(None, 0, 0))(
            params, states_norm, actions_norm
        )

        # Compute MSE loss in normalized space
        mse_loss = jnp.mean(jnp.square(pred_outputs_norm - targets_norm))

        # Denormalize predictions for metrics
        pred_outputs = jax.vmap(
            denormalize_output, in_axes=(0, None, None)
        )(pred_outputs_norm, output_mean, output_std)

        # Add L2 regularization
        if weight_decay > 0:
            l2_loss = sum(
                jnp.sum(jnp.square(p))
                for p in jax.tree_util.tree_leaves(params)
            )
            total_loss = mse_loss + weight_decay * l2_loss
        else:
            total_loss = mse_loss
            l2_loss = 0.0

        # Compute MAE in original space for interpretability
        mae = jnp.mean(jnp.abs(pred_outputs - targets))

        metrics = {
            "loss": total_loss,
            "mse": mse_loss,
            "mae": mae,
            "l2_loss": l2_loss,
        }

        return total_loss, metrics

    return loss_fn


def train_step(
    state: TrainingState,
    batch: Tuple[jax.Array, jax.Array, jax.Array],
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    grad_clip: float = None,
) -> Tuple[TrainingState, dict]:
    """Perform a single training step."""
    states, actions, targets = batch

    # Compute loss and gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, states, actions, targets
    )

    # Optionally clip gradients
    if grad_clip is not None:
        grads = optax.clip_by_global_norm(grad_clip)(grads)

    # Update parameters
    updates, new_opt_state = optimizer.update(
        grads, state.opt_state, state.params
    )
    new_params = optax.apply_updates(state.params, updates)

    # Update training state
    new_state = state.replace(params=new_params, opt_state=new_opt_state)

    return new_state, metrics


def prepare_epoch_batches(
    dataset,
    targets: jax.Array,
    batch_size: int,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare all batches for an epoch by pre-shuffling.

    Args:
        dataset: The dynamics dataset.
        targets: Training targets (state deltas or accelerations).
        batch_size: Size of each batch.
        rng: Random number generator key.

    Returns:
        Tuple of (states_batches, actions_batches, targets_batches, rngs_batches) where
        each has shape (num_batches, batch_size, ...).
    """
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Split RNG for shuffling and noise
    shuffle_rng, noise_rng = jax.random.split(rng)

    # Shuffle indices once for the entire epoch
    indices = jax.random.permutation(shuffle_rng, num_samples)

    # Pad dataset to be divisible by batch_size
    remainder = num_samples % batch_size
    if remainder > 0:
        padding_size = batch_size - remainder
        padding_indices = jnp.tile(indices[-1:], (padding_size,))
        indices = jnp.concatenate([indices, padding_indices])

    # Reshape into batches
    indices_reshaped = indices.reshape(num_batches, batch_size)

    # Use advanced indexing to get all batches at once
    states_batches = dataset.states[indices_reshaped]
    actions_batches = dataset.actions[indices_reshaped]
    targets_batches = targets[indices_reshaped]

    # Generate RNG keys for each batch (for noise augmentation)
    rngs_batches = jax.random.split(noise_rng, num_batches)

    return states_batches, actions_batches, targets_batches, rngs_batches


def create_epoch_train_fn(
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    grad_clip: float = None,
) -> Callable:
    """Create a JIT-compiled epoch training function using scan.

    Args:
        loss_fn: Loss function to optimize.
        optimizer: Optax optimizer.
        grad_clip: Maximum gradient norm (None for no clipping).

    Returns:
        JIT-compiled epoch training function.
    """

    def train_step_inner(
        state: TrainingState,
        batch: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> Tuple[TrainingState, dict]:
        """Single training step."""
        states, actions, targets, rng = batch

        # Compute loss and gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, states, actions, targets, rng
        )

        # Optionally clip gradients
        if grad_clip is not None:
            grads = optax.clip_by_global_norm(grad_clip)(grads)

        # Update parameters
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, state.params
        )
        new_params = optax.apply_updates(state.params, updates)

        # Update training state
        new_state = state.replace(params=new_params, opt_state=new_opt_state)

        return new_state, metrics

    def epoch_train_fn(
        state: TrainingState,
        batches: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> Tuple[TrainingState, dict]:
        """Train for one epoch using scan."""
        states_batch, actions_batch, targets_batch, rngs_batch = batches

        def scan_fn(state, batch_data):
            """Scan function for processing batches."""
            states, actions, targets, rng = batch_data
            new_state, metrics = train_step_inner(
                state, (states, actions, targets, rng)
            )
            return new_state, metrics

        # Use scan to process all batches
        final_state, metrics_list = jax.lax.scan(
            scan_fn,
            state,
            (states_batch, actions_batch, targets_batch, rngs_batch),
        )

        # Average metrics across batches
        avg_metrics = {k: jnp.mean(v) for k, v in metrics_list.items()}

        return final_state, avg_metrics

    # JIT compile the entire epoch function
    return jax.jit(epoch_train_fn, donate_argnums=(0,))


def _print_evaluation_summary(
    model: BaseDynamicsModel,
    params: DynamicsModelParams,
    val_dataset: DynamicsDataset,
    env,
    eval_num_samples: int,
    eval_rollout_length: int,
    eval_num_rollouts: int,
    rng: jax.Array,
    tb_writer: Optional[SummaryWriter] = None,
    config: Optional[TrainingConfig] = None,
):
    """Print evaluation summary after training."""
    from dynax.evaluation import (
        create_rollout_fn,
        evaluate_single_step_dataset,
    )
    from mujoco import mjx

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Single-step evaluation
    print("\nSingle-Step Prediction:")
    single_step_results = evaluate_single_step_dataset(
        model, params, val_dataset, num_samples=eval_num_samples, rng=rng
    )
    print(f"  MAE:  {single_step_results['mae_mean']:.6f}")
    print(f"  RMSE: {single_step_results['rmse_mean']:.6f}")

    # Log to TensorBoard
    if tb_writer is not None:
        tb_writer.add_scalar("eval/single_step_mae", float(single_step_results["mae_mean"]), 0)
        tb_writer.add_scalar("eval/single_step_rmse", float(single_step_results["rmse_mean"]), 0)

    # Multi-step rollout evaluation (if env provided)
    if env is not None:
        from dynax.evaluation import evaluate_rollouts_batch

        print("\nMulti-Step Rollout:")
        neural_rollout = create_rollout_fn(model, params)

        # Pre-generate all initial states and actions in batch
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, eval_num_rollouts)

        # Vectorized reset function
        def reset_single(r):
            data = mjx.make_data(env.model)
            data = env.reset(data, r)
            data = mjx.forward(env.model, data)
            return jnp.concatenate([data.qpos, data.qvel])

        initial_states = jax.vmap(reset_single)(reset_rngs)

        # Pre-generate all random actions
        rng, action_rng = jax.random.split(rng)
        actions_batch = jax.random.uniform(
            action_rng,
            (eval_num_rollouts, eval_rollout_length, env.model.nu),
            minval=env.action_min,
            maxval=env.action_max,
        )

        # Evaluate all rollouts in parallel (with trajectories for metrics)
        rollout_results = evaluate_rollouts_batch(
            env.model,
            neural_rollout,
            initial_states,
            actions_batch,
            return_trajectories=True,
        )

        # Compute std dev directly from trajectories (no re-computation!)
        true_trajs = rollout_results["true_trajectories"]
        pred_trajs = rollout_results["pred_trajectories"]
        final_errors = jnp.mean(jnp.abs(pred_trajs[:, -1, :] - true_trajs[:, -1, :]), axis=1)
        std_final_mae = float(jnp.std(final_errors))

        print(f"  Final MAE:  {rollout_results['final_mae']:.6f} ± {std_final_mae:.6f}")
        print(f"  Final RMSE: {rollout_results['final_rmse']:.6f}")
        print(f"  (averaged over {eval_num_rollouts} rollouts of length {eval_rollout_length})")

        # Log to TensorBoard
        if tb_writer is not None:
            from dynax.evaluation import plot_trajectories

            tb_writer.add_scalar("eval/rollout_final_mae", float(rollout_results["final_mae"]), 0)
            tb_writer.add_scalar("eval/rollout_final_mae_std", float(std_final_mae), 0)
            tb_writer.add_scalar("eval/rollout_final_rmse", float(rollout_results["final_rmse"]), 0)
            # Log error over time (sample every 10th step to avoid too many files)
            mae_over_time = rollout_results["mae_over_time"]
            rmse_over_time = rollout_results["rmse_over_time"]
            # Log sampled points (every 10th step) to avoid creating too many files
            sample_every = max(1, len(mae_over_time) // 20)  # ~20 points max
            for t in range(0, len(mae_over_time), sample_every):
                tb_writer.add_scalar("eval/rollout_mae_over_time", float(mae_over_time[t]), t)
                tb_writer.add_scalar("eval/rollout_rmse_over_time", float(rmse_over_time[t]), t)

            # Create and log trajectory plots (optimized for speed)
            true_trajs = rollout_results["true_trajectories"]
            pred_trajs = rollout_results["pred_trajectories"]
            state_dim = true_trajs.shape[-1]
            nq = env.model.nq

            # Create trajectory plots (reduced for speed)
            plot_array = plot_trajectories(
                true_trajs,
                pred_trajs,
                state_dim=state_dim,
                nq=nq,
                num_plots=2,  # Reduced for speed
                save_path=Path(tb_writer.logdir) / "trajectory_plots.png",
            )

            # Log to TensorBoard (add_image expects CHW format)
            tb_writer.add_image(
                "eval/trajectories", plot_array.transpose(2, 0, 1), 0, dataformats="CHW"
            )

            # Optionally render videos (optimized but still CPU-bound)
            if config is not None and config.render_videos:
                from dynax.evaluation import render_trajectory_video

                num_videos = min(1, len(true_trajs))  # Render only 1 video for speed

                for vid_idx in range(num_videos):
                    true_traj = true_trajs[vid_idx]
                    pred_traj = pred_trajs[vid_idx]

                    # Save video file (high quality)
                    video_path = Path(tb_writer.logdir) / f"trajectory_video_{vid_idx}.mp4"
                    video_frames = render_trajectory_video(
                        env,
                        true_traj,
                        pred_traj,
                        fps=30,  # Higher FPS for smoother playback
                        save_path=video_path,
                        max_frames=200,  # More frames for longer videos
                    )

                    # Also log to TensorBoard (convert HWC to CHW)
                    video_chw = video_frames.transpose(0, 3, 1, 2)  # T, H, W, C -> T, C, H, W
                    tb_writer.add_video(
                        f"eval/trajectory_video_{vid_idx}",
                        video_chw[None, :],  # Add batch dimension: (1, T, C, H, W)
                        0,
                        fps=30,  # Match video FPS
                    )

    print("\n" + "=" * 60)

    return {
        "single_step": single_step_results,
        "rollout": rollout_results if env is not None else None,
    }


def train_dynamics_model(
    model: BaseDynamicsModel,
    train_dataset: DynamicsDataset,
    val_dataset: DynamicsDataset,
    config: TrainingConfig,
    rng: jax.Array,
    verbose: bool = True,
    env=None,
    eval_num_samples: int = 1000,
    eval_rollout_length: int = 100,
    eval_num_rollouts: int = 10,  # Reduced default for faster evaluation
) -> DynamicsModelParams:
    """Train a dynamics model on collected data.

    Args:
        model: Dynamics model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Training configuration.
        rng: Random number generator key.
        verbose: Whether to print training progress.
        env: Optional environment for rollout evaluation.
        eval_num_samples: Number of samples for single-step evaluation.
        eval_rollout_length: Length of evaluation rollouts.
        eval_num_rollouts: Number of evaluation rollouts.

    Returns:
        Trained model parameters.
    """
    # Let the model specify what it predicts
    train_targets = model.prepare_training_targets(train_dataset)
    val_targets = model.prepare_training_targets(val_dataset)

    # Compute normalization statistics
    state_mean, state_std, output_mean, output_std = (
        compute_normalization_stats(train_dataset.states, train_targets)
    )

    # Compute action normalization statistics
    action_mean, action_std = compute_action_normalization_stats(
        train_dataset.actions
    )

    # Initialize model parameters
    rng, subrng = jax.random.split(rng)
    dummy_state = jnp.zeros(model.state_dim)
    dummy_action = jnp.zeros(model.action_dim)
    params = model.init(subrng, dummy_state, dummy_action)

    # Create optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    # Create loss function (use custom or default)
    if config.loss_fn is not None:
        loss_fn = config.loss_fn
    else:
        loss_fn = create_loss_fn(
            model,
            state_mean,
            state_std,
            action_mean,
            action_std,
            output_mean,
            output_std,
            config.weight_decay,
            config.noise_std,
        )

    # Initialize training state
    train_state = TrainingState(
        params=params,
        opt_state=opt_state,
        epoch=0,
        best_val_loss=float("inf"),
    )

    # Create JIT-compiled epoch training function
    epoch_train_fn = create_epoch_train_fn(
        loss_fn, optimizer, config.grad_clip
    )

    # JIT-compiled validation function (no noise augmentation for validation)
    @jax.jit
    def validate(params):
        _, metrics = loss_fn(params, val_dataset.states, val_dataset.actions, val_targets, None)
        return metrics

    # Set up TensorBoard logging
    tb_writer = None
    log_dir_path = None
    if config.log_dir is not None:
        log_dir_path = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard logging to {log_dir_path}")
        tb_writer = SummaryWriter(str(log_dir_path))
        # Store logdir for later use
        tb_writer.logdir = str(log_dir_path)

    # Training loop
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        # Prepare all batches for this epoch (pre-shuffled)
        rng, epoch_rng = jax.random.split(rng)
        batches = prepare_epoch_batches(
            train_dataset, train_targets, config.batch_size, epoch_rng
        )

        # Train entire epoch in one compiled call using scan
        train_state, train_metrics = epoch_train_fn(train_state, batches)

        # Validation
        val_metrics = validate(train_state.params)

        # Update best validation loss
        if val_metrics["loss"] < train_state.best_val_loss:
            train_state = train_state.replace(
                best_val_loss=val_metrics["loss"]
            )

        # Log to TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", float(train_metrics["loss"]), epoch)
            tb_writer.add_scalar("train/mse", float(train_metrics["mse"]), epoch)
            tb_writer.add_scalar("train/mae", float(train_metrics["mae"]), epoch)
            if "l2_loss" in train_metrics:
                tb_writer.add_scalar("train/l2_loss", float(train_metrics["l2_loss"]), epoch)

            tb_writer.add_scalar("val/loss", float(val_metrics["loss"]), epoch)
            tb_writer.add_scalar("val/mse", float(val_metrics["mse"]), epoch)
            tb_writer.add_scalar("val/mae", float(val_metrics["mae"]), epoch)

            tb_writer.add_scalar("train/best_val_loss", float(train_state.best_val_loss), epoch)
            tb_writer.add_scalar("train/learning_rate", config.learning_rate, epoch)

            epoch_time = time.time() - epoch_start_time
            tb_writer.add_scalar("train/epoch_time", epoch_time, epoch)

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == config.num_epochs - 1):
            print(
                f"Epoch {epoch:3d}: "
                f"Train Loss = {train_metrics['loss']:.6f}, "
                f"Val Loss = {val_metrics['loss']:.6f}, "
                f"Val MAE = {val_metrics['mae']:.6f}"
            )

        train_state = train_state.replace(epoch=epoch + 1)

    # Create final model parameters
    trained_params = DynamicsModelParams(
        network_params=train_state.params,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        output_mean=output_mean,
        output_std=output_std,
    )

    # Evaluation
    if verbose:
        _print_evaluation_summary(
            model,
            trained_params,
            val_dataset,
            env,
            eval_num_samples,
            eval_rollout_length,
            eval_num_rollouts,
            rng,
            tb_writer,
            config,
        )

    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()

    return trained_params

