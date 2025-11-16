"""Base classes for neural dynamics models."""

from abc import abstractmethod

import jax
from flax import linen as nn
from flax.struct import dataclass

from dynax.envs import Env


@dataclass
class DynamicsModelParams:
    """Parameters for a learned dynamics model.

    Attributes:
        network_params: Neural network parameters.
        state_mean: Mean of state features for normalization.
        state_std: Std deviation of state features for normalization.
        action_mean: Mean of actions for normalization.
        action_std: Std deviation of actions for normalization.
        output_mean: Mean of model outputs (accelerations).
        output_std: Std deviation of model outputs.
    """

    network_params: dict
    state_mean: jax.Array
    state_std: jax.Array
    action_mean: jax.Array
    action_std: jax.Array
    output_mean: jax.Array
    output_std: jax.Array


class BaseDynamicsModel(nn.Module):
    """Abstract base class for neural dynamics models.

    Attributes:
        env: Environment providing model dimensions and timestep.
        history_length: Number of past (state, action) pairs to use as input.
            Default is 1 (single step, backward compatible).
    """

    env: Env
    history_length: int = 1

    @property
    def state_dim(self) -> int:
        """State dimension: nq + nv."""
        return self.env.model.nq + self.env.model.nv

    @property
    def action_dim(self) -> int:
        """Action dimension: nu (number of actuators)."""
        return self.env.model.nu

    @property
    def dt(self) -> float:
        """Timestep from environment."""
        return self.env.dt

    @abstractmethod
    def __call__(
        self, states: jax.Array, actions: jax.Array
    ) -> jax.Array:
        """Predict the model output given state-action history.

        Args:
            states: State history, shape (history_length, state_dim).
            actions: Action history, shape (history_length, action_dim).

        Returns:
            Predicted output (e.g., next state delta), shape (output_dim,).
        """
        pass

    def prepare_training_targets(self, dataset) -> jax.Array:
        """Prepare training targets from a dataset.

        Override this method to specify what your model predicts.
        Default: state deltas (next_state - current_state).

        Args:
            dataset: DynamicsDataset with states, actions, next_states, etc.
                States may be windowed (N, history_length, state_dim) or
                single-step (N, state_dim).

        Returns:
            Training targets with shape (N, output_dim).
        """
        # Handle windowed states: use most recent state from history
        if dataset.states.ndim == 3:
            current_states = dataset.states[:, -1, :]  # Most recent state
        else:
            current_states = dataset.states
        return dataset.next_states - current_states

    def step(
        self,
        params: DynamicsModelParams,
        states: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """Predict next state using the trained model.

        Default implementation for residual/delta-based models.
        Override this method for physics-informed models (like Euler).

        Args:
            params: Model parameters.
            states: State history, shape (history_length, state_dim).
            actions: Action history, shape (history_length, action_dim).

        Returns:
            Next state, shape (state_dim,).
        """
        from dynax.utils.normalization import (
            denormalize_output,
            normalize_action,
            normalize_state,
        )

        # Normalize states and actions (vmap over history dimension)
        states_norm = jax.vmap(normalize_state, in_axes=(0, None, None))(
            states, params.state_mean, params.state_std
        )
        actions_norm = jax.vmap(normalize_action, in_axes=(0, None, None))(
            actions, params.action_mean, params.action_std
        )

        # Predict normalized output
        output_norm = self.apply(
            params.network_params, states_norm, actions_norm
        )

        # Denormalize output
        output = denormalize_output(
            output_norm, params.output_mean, params.output_std
        )

        # Default: residual dynamics (next_state = current_state + output)
        # Use the most recent state from history
        return states[-1] + output

