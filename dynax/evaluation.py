"""Optimized evaluation utilities for dynamics models."""

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from dynax.base import BaseDynamicsModel, DynamicsModelParams
from dynax.envs import Env
from dynax.utils.data import DynamicsDataset, extract_state_features


def evaluate_single_step(
    model: BaseDynamicsModel,
    params: DynamicsModelParams,
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
    model: BaseDynamicsModel,
    params: DynamicsModelParams,
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
    model: BaseDynamicsModel,
    params: DynamicsModelParams,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Create a JIT-compiled rollout function.

    Args:
        model: Dynamics model.
        params: Model parameters.

    Returns:
        JIT-compiled function that takes (initial_state, actions) and returns
        trajectory of states.
    """

    @jax.jit
    def rollout_fn(initial_state: jax.Array, actions: jax.Array) -> jax.Array:
        """Roll out trajectory using neural dynamics model.

        Args:
            initial_state: Initial state, shape (state_dim,).
            actions: Actions sequence, shape (T, action_dim).

        Returns:
            State trajectory, shape (T+1, state_dim).
        """

        def step_fn(state, action):
            next_state = model.step(params, state, action)
            return next_state, next_state

        _, states = jax.lax.scan(step_fn, initial_state, actions)
        return jnp.concatenate([initial_state[None, :], states], axis=0)

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
    initial_state: jax.Array,
    initial_data: mjx.Data,
    actions: jax.Array,
) -> Dict[str, jax.Array]:
    """Evaluate multi-step rollout accuracy.

    Args:
        neural_rollout_fn: Neural model rollout function.
        true_rollout_fn: True dynamics rollout function.
        initial_state: Initial state features, shape (state_dim,).
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
    pred_states = neural_rollout_fn(initial_state, actions)

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
) -> Dict[str, jax.Array]:
    """Evaluate multiple rollouts in parallel using vmap.

    Args:
        model: MJX model for true dynamics.
        neural_rollout_fn: Neural model rollout function.
        initial_states: Initial state features, shape (N, state_dim).
        actions_batch: Actions sequences, shape (N, T, action_dim).

    Returns:
        Dictionary with metrics averaged over rollouts:
        - mae_over_time: Mean absolute error over time, shape (T+1,).
        - rmse_over_time: Root mean squared error over time, shape (T+1,).
        - final_mae: Final step MAE (scalar).
        - final_rmse: Final step RMSE (scalar).
    """

    def single_rollout_eval(initial_state, actions):
        """Evaluate single rollout (vmappable)."""
        # Create initial MJX data from state
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
        pred_states = neural_rollout_fn(initial_state, actions)

        # Compute errors
        errors = jnp.abs(pred_states - true_states)
        squared_errors = jnp.square(pred_states - true_states)

        mae_over_time = jnp.mean(errors, axis=1)
        rmse_over_time = jnp.sqrt(jnp.mean(squared_errors, axis=1))

        return {
            "mae_over_time": mae_over_time,
            "rmse_over_time": rmse_over_time,
            "final_mae": mae_over_time[-1],
            "final_rmse": rmse_over_time[-1],
        }

    # Vmap over all rollouts in parallel
    results = jax.vmap(single_rollout_eval)(initial_states, actions_batch)

    # Average metrics
    return {
        "mae_over_time": jnp.mean(results["mae_over_time"], axis=0),
        "rmse_over_time": jnp.mean(results["rmse_over_time"], axis=0),
        "final_mae": jnp.mean(results["final_mae"]),
        "final_rmse": jnp.mean(results["final_rmse"]),
    }

