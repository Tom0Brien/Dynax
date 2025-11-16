"""Data collection and dataset utilities."""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx


@dataclass
class DynamicsDataset:
    """Dataset for training dynamics models.

    Attributes:
        states: State observations, shape (N, state_dim).
        actions: Actions taken, shape (N, action_dim).
        next_states: Next states, shape (N, state_dim).
        accelerations: Accelerations (for physics-informed models),
            shape (N, nv).
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        nq: Number of position dimensions.
        nv: Number of velocity dimensions.
        dt: Timestep used for data collection.
    """

    states: jax.Array
    actions: jax.Array
    next_states: jax.Array
    accelerations: jax.Array
    state_dim: int
    action_dim: int
    nq: int
    nv: int
    dt: float

    def __len__(self) -> int:
        return len(self.states)


def extract_state_features(data: mjx.Data) -> jax.Array:
    """Extract state features from MJX data.

    Args:
        data: MJX data object.

    Returns:
        State features as [qpos, qvel].
    """
    return jnp.concatenate([data.qpos, data.qvel])


def collect_random_rollouts(
    model: mjx.Model,
    num_rollouts: int,
    rollout_length: int,
    action_min: jax.Array,
    action_max: jax.Array,
    rng: jax.Array,
    initial_state_sampler: Optional[Callable] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Collect rollouts with random actions using JAX parallelization.

    This function uses vmap to parallelize across rollouts and scan to
    handle sequential steps, resulting in massive speedups on GPU.

    Args:
        model: MJX model.
        num_rollouts: Number of rollouts to collect.
        rollout_length: Length of each rollout.
        action_min: Minimum action values.
        action_max: Maximum action values.
        rng: Random number generator key.
        initial_state_sampler: Optional function to sample initial states.

    Returns:
        Tuple of (states, actions, next_states, accelerations).
    """

    def single_rollout(rng_key):
        """Execute a single rollout with pre-generated actions."""
        # Split RNG for initialization and actions
        init_rng, action_rng = jax.random.split(rng_key)

        # Initialize state
        if initial_state_sampler is not None:
            initial_state = initial_state_sampler(model, init_rng)
        else:
            initial_state = mjx.make_data(model)
            initial_state = mjx.forward(model, initial_state)

        # Pre-generate all random actions for this rollout
        actions = jax.random.uniform(
            action_rng,
            (rollout_length, model.nu),
            minval=action_min,
            maxval=action_max,
        )

        def step_fn(state, action):
            """Single simulation step."""
            # Extract current state features and acceleration
            current_state = extract_state_features(state)
            current_accel = state.qacc

            # Step simulation
            state = state.replace(ctrl=action)
            next_state = mjx.step(model, state)
            next_state_features = extract_state_features(next_state)

            # Return next state and transition data
            return next_state, (
                current_state,
                action,
                next_state_features,
                current_accel,
            )

        # Use scan to efficiently process sequential steps
        _, transitions = jax.lax.scan(step_fn, initial_state, actions)

        return transitions

    # JIT compile the single rollout function
    jit_single_rollout = jax.jit(single_rollout)

    # Generate RNG keys for all rollouts
    rng_keys = jax.random.split(rng, num_rollouts)

    # Vectorize across rollouts for parallel execution
    all_transitions = jax.vmap(jit_single_rollout)(rng_keys)

    # Unpack and reshape: (num_rollouts, rollout_length, ...) -> (N, ...)
    states = all_transitions[0].reshape(-1, model.nq + model.nv)
    actions = all_transitions[1].reshape(-1, model.nu)
    next_states = all_transitions[2].reshape(-1, model.nq + model.nv)
    accelerations = all_transitions[3].reshape(-1, model.nv)

    return states, actions, next_states, accelerations


def create_dataset(
    model: mjx.Model,
    states: jax.Array,
    actions: jax.Array,
    next_states: jax.Array,
    accelerations: jax.Array,
    dt: float,
) -> DynamicsDataset:
    """Create a DynamicsDataset from collected data.

    Args:
        model: MJX model.
        states: State observations.
        actions: Actions taken.
        next_states: Next states.
        accelerations: Accelerations.
        dt: Timestep.

    Returns:
        DynamicsDataset object.
    """
    state_dim = model.nq + model.nv
    action_dim = model.nu

    return DynamicsDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        accelerations=accelerations,
        state_dim=state_dim,
        action_dim=action_dim,
        nq=model.nq,
        nv=model.nv,
        dt=dt,
    )


def split_dataset(
    dataset: DynamicsDataset,
    train_ratio: float = 0.8,
    rng: Optional[jax.Array] = None,
) -> Tuple[DynamicsDataset, DynamicsDataset]:
    """Split dataset into training and validation sets.

    Args:
        dataset: Dataset to split.
        train_ratio: Fraction of data to use for training.
        rng: Random number generator key for shuffling.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    num_samples = len(dataset)
    num_train = int(num_samples * train_ratio)

    # Optionally shuffle
    if rng is not None:
        indices = jax.random.permutation(rng, num_samples)
    else:
        indices = jnp.arange(num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = DynamicsDataset(
        states=dataset.states[train_indices],
        actions=dataset.actions[train_indices],
        next_states=dataset.next_states[train_indices],
        accelerations=dataset.accelerations[train_indices],
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        nq=dataset.nq,
        nv=dataset.nv,
        dt=dataset.dt,
    )

    val_dataset = DynamicsDataset(
        states=dataset.states[val_indices],
        actions=dataset.actions[val_indices],
        next_states=dataset.next_states[val_indices],
        accelerations=dataset.accelerations[val_indices],
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        nq=dataset.nq,
        nv=dataset.nv,
        dt=dataset.dt,
    )

    return train_dataset, val_dataset

