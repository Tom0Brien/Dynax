"""Training utilities for neural network dynamics models."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass

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
        learning_rate: Learning rate for optimizer.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        weight_decay: L2 regularization weight.
        grad_clip: Maximum gradient norm (None for no clipping).
    """

    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 100
    weight_decay: float = 1e-4
    grad_clip: float = None


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
) -> Callable:
    """Create a loss function for training the dynamics model."""

    def loss_fn(
        params: dict,
        states: jax.Array,
        actions: jax.Array,
        targets: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        """Compute MSE loss between predicted and actual outputs."""
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
    batch_size: int,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Prepare all batches for an epoch by pre-shuffling.

    Args:
        dataset: The dynamics dataset.
        batch_size: Size of each batch.
        rng: Random number generator key.

    Returns:
        Tuple of (states_batches, actions_batches, targets_batches) where
        each has shape (num_batches, batch_size, ...).
    """
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Shuffle indices once for the entire epoch
    indices = jax.random.permutation(rng, num_samples)

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
    targets_batches = dataset.accelerations[indices_reshaped]

    return states_batches, actions_batches, targets_batches


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
        batch: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> Tuple[TrainingState, dict]:
        """Single training step."""
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

    def epoch_train_fn(
        state: TrainingState,
        batches: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> Tuple[TrainingState, dict]:
        """Train for one epoch using scan."""
        states_batch, actions_batch, targets_batch = batches

        def scan_fn(state, batch_data):
            """Scan function for processing batches."""
            states, actions, targets = batch_data
            new_state, metrics = train_step_inner(
                state, (states, actions, targets)
            )
            return new_state, metrics

        # Use scan to process all batches
        final_state, metrics_list = jax.lax.scan(
            scan_fn,
            state,
            (states_batch, actions_batch, targets_batch),
        )

        # Average metrics across batches
        avg_metrics = {k: jnp.mean(v) for k, v in metrics_list.items()}

        return final_state, avg_metrics

    # JIT compile the entire epoch function
    return jax.jit(epoch_train_fn, donate_argnums=(0,))


def train_dynamics_model(
    model: BaseDynamicsModel,
    train_dataset: DynamicsDataset,
    val_dataset: DynamicsDataset,
    config: TrainingConfig,
    rng: jax.Array,
    verbose: bool = True,
) -> DynamicsModelParams:
    """Train a dynamics model on collected data."""
    # Compute normalization statistics
    state_mean, state_std, output_mean, output_std = (
        compute_normalization_stats(
            train_dataset.states, train_dataset.accelerations
        )
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

    # Create loss function
    loss_fn = create_loss_fn(
        model,
        state_mean,
        state_std,
        action_mean,
        action_std,
        output_mean,
        output_std,
        config.weight_decay,
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

    # JIT-compiled validation function
    @jax.jit
    def validate(params):
        _, metrics = loss_fn(
            params,
            val_dataset.states,
            val_dataset.actions,
            val_dataset.accelerations,
        )
        return metrics

    # Training loop
    for epoch in range(config.num_epochs):
        # Prepare all batches for this epoch (pre-shuffled)
        rng, epoch_rng = jax.random.split(rng)
        batches = prepare_epoch_batches(
            train_dataset, config.batch_size, epoch_rng
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

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == config.num_epochs - 1):
            print(
                f"Epoch {epoch:3d}: "
                f"Train Loss = {train_metrics['loss']:.6f}, "
                f"Val Loss = {val_metrics['loss']:.6f}, "
                f"Val MAE = {val_metrics['mae']:.6f}"
            )

        train_state = train_state.replace(epoch=epoch + 1)

    # Return trained parameters with normalization stats
    return DynamicsModelParams(
        network_params=train_state.params,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        output_mean=output_mean,
        output_std=output_std,
        dt=train_dataset.dt,
    )

