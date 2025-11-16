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
    """

    env: Env

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
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Predict the model output given current state and action."""
        pass

    def prepare_training_targets(self, dataset) -> jax.Array:
        """Prepare training targets from a dataset.

        Override this method to specify what your model predicts.
        Default: state deltas (next_state - state).

        Args:
            dataset: DynamicsDataset with states, actions, next_states, etc.

        Returns:
            Training targets with shape (N, output_dim).
        """
        return dataset.next_states - dataset.states

    def step(
        self,
        params: DynamicsModelParams,
        state: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        """Predict next state using the trained model.

        Default implementation for residual/delta-based models.
        Override this method for physics-informed models (like Euler).
        """
        from dynax.utils.normalization import (
            denormalize_output,
            normalize_action,
            normalize_state,
        )

        # Normalize state and action
        state_norm = normalize_state(
            state, params.state_mean, params.state_std
        )
        action_norm = normalize_action(
            action, params.action_mean, params.action_std
        )

        # Predict normalized output
        output_norm = self.apply(
            params.network_params, state_norm, action_norm
        )

        # Denormalize output
        output = denormalize_output(
            output_norm, params.output_mean, params.output_std
        )

        # Default: residual dynamics (next_state = state + output)
        return state + output

