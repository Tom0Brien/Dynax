"""Normalization utilities for states, actions, and outputs."""

from typing import Tuple

import jax.numpy as jnp


def compute_normalization_stats(
    states: jnp.ndarray,
    targets: jnp.ndarray,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute normalization statistics for states and targets.

    Args:
        states: State features, shape (N, state_dim).
        targets: Target outputs (e.g., accelerations), shape (N, output_dim).
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (state_mean, state_std, target_mean, target_std).
    """
    state_mean = jnp.mean(states, axis=0)
    state_std = jnp.std(states, axis=0) + eps

    target_mean = jnp.mean(targets, axis=0)
    target_std = jnp.std(targets, axis=0) + eps

    return state_mean, state_std, target_mean, target_std


def compute_action_normalization_stats(
    actions: jnp.ndarray,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute normalization statistics for actions.

    Args:
        actions: Actions, shape (N, action_dim).
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (action_mean, action_std).
    """
    action_mean = jnp.mean(actions, axis=0)
    action_std = jnp.std(actions, axis=0) + eps

    return action_mean, action_std


def normalize_state(
    state: jnp.ndarray,
    state_mean: jnp.ndarray,
    state_std: jnp.ndarray,
) -> jnp.ndarray:
    """Normalize state features using precomputed statistics."""
    return (state - state_mean) / state_std


def normalize_action(
    action: jnp.ndarray,
    action_mean: jnp.ndarray,
    action_std: jnp.ndarray,
) -> jnp.ndarray:
    """Normalize actions using precomputed statistics."""
    return (action - action_mean) / action_std


def denormalize_output(
    output_norm: jnp.ndarray,
    output_mean: jnp.ndarray,
    output_std: jnp.ndarray,
) -> jnp.ndarray:
    """Denormalize model outputs to original scale."""
    return output_norm * output_std + output_mean

