"""Base classes for neural dynamics models."""

import pickle
from abc import abstractmethod
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.struct import dataclass

from dynax.envs import Env


@dataclass
class NeuralModelParams:
    """Parameters for a learned neural model.

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


class BaseNeuralModel(nn.Module):
    """Abstract base class for neural models.

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
        self, states: jax.Array, actions: jax.Array, training: bool = False
    ) -> jax.Array:
        """Predict the model output given state-action history.

        Args:
            states: State history, shape (history_length, state_dim).
            actions: Action history, shape (history_length, action_dim).
            training: Whether in training mode (affects dropout, etc.).

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
        params: NeuralModelParams,
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

        # Predict normalized output (training=False for inference)
        output_norm = self.apply(
            params.network_params, states_norm, actions_norm, training=False
        )

        # Denormalize output
        output = denormalize_output(
            output_norm, params.output_mean, params.output_std
        )

        # Default: residual dynamics (next_state = current_state + output)
        # Use the most recent state from history
        return states[-1] + output

    def save_model(self, params: NeuralModelParams, path: str | Path) -> None:
        """Save the model parameters to disk.

        Args:
            params: Model parameters to save.
            path: Path to save the model parameters (uses .pkl extension
                if not provided).
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pkl")

        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert JAX arrays to numpy for serialization
        def to_numpy(x):
            """Recursively convert JAX arrays to numpy arrays."""
            if isinstance(x, (jax.Array, jnp.ndarray)):
                return np.array(x)
            elif isinstance(x, dict):
                return {k: to_numpy(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                return type(x)(to_numpy(v) for v in x)
            return x

        params_dict = {
            "network_params": to_numpy(params.network_params),
            "state_mean": to_numpy(params.state_mean),
            "state_std": to_numpy(params.state_std),
            "action_mean": to_numpy(params.action_mean),
            "action_std": to_numpy(params.action_std),
            "output_mean": to_numpy(params.output_mean),
            "output_std": to_numpy(params.output_std),
        }

        with open(path, "wb") as f:
            pickle.dump(params_dict, f)

    def load_model(self, path: str | Path) -> NeuralModelParams:
        """Load model parameters from disk.

        Args:
            path: Path to the saved model parameters file.

        Returns:
            Loaded NeuralModelParams.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            params_dict = pickle.load(f)

        # Convert numpy arrays back to JAX arrays
        def to_jax(x):
            """Recursively convert numpy arrays to JAX arrays."""
            if isinstance(x, np.ndarray):
                return jnp.array(x)
            elif isinstance(x, dict):
                return {k: to_jax(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                return type(x)(to_jax(v) for v in x)
            return x

        return NeuralModelParams(
            network_params=to_jax(params_dict["network_params"]),
            state_mean=jnp.array(params_dict["state_mean"]),
            state_std=jnp.array(params_dict["state_std"]),
            action_mean=jnp.array(params_dict["action_mean"]),
            action_std=jnp.array(params_dict["action_std"]),
            output_mean=jnp.array(params_dict["output_mean"]),
            output_std=jnp.array(params_dict["output_std"]),
        )

