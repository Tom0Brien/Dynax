"""Data collection and dataset utilities."""

import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from mujoco import mjx

from dynax.utils.controllers import Controller


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


def collect_rollouts(
    model: mjx.Model,
    num_rollouts: int,
    rollout_length: int,
    action_min: jax.Array,
    action_max: jax.Array,
    rng: jax.Array,
    controller: Optional[Controller] = None,
    initial_state_sampler: Optional[Callable] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Collect rollouts using either random actions or a controller.

    This function uses vmap to parallelize across rollouts and scan to
    handle sequential steps, resulting in massive speedups on GPU.

    Args:
        model: MJX model.
        num_rollouts: Number of rollouts to collect.
        rollout_length: Length of each rollout.
        action_min: Minimum action values (used for random actions).
        action_max: Maximum action values (used for random actions).
        rng: Random number generator key.
        controller: Optional controller for controlled rollouts. If None, uses
            random actions.
        initial_state_sampler: Optional function to sample initial states.

    Returns:
        Tuple of (states, actions, next_states, accelerations).
    """
    if controller is None:
        # Random rollouts: pre-generate all actions
        def single_rollout(rng_key):
            """Execute a single rollout with pre-generated random actions."""
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
                current_state = extract_state_features(state)
                current_accel = state.qacc

                state = state.replace(ctrl=action)
                next_state = mjx.step(model, state)
                next_state_features = extract_state_features(next_state)

                return next_state, (
                    current_state,
                    action,
                    next_state_features,
                    current_accel,
                )

            _, transitions = jax.lax.scan(step_fn, initial_state, actions)
            return transitions

        jit_single_rollout = jax.jit(single_rollout)
        rng_keys = jax.random.split(rng, num_rollouts)
        all_transitions = jax.vmap(jit_single_rollout)(rng_keys)

    else:
        # Controlled rollouts: controller generates actions on-the-fly
        def single_rollout(rng_key):
            """Execute a single rollout with controller."""
            init_rng, ctrl_rng = jax.random.split(rng_key)

            # Initialize state
            if initial_state_sampler is not None:
                initial_state = initial_state_sampler(model, init_rng)
            else:
                initial_state = mjx.make_data(model)
                initial_state = mjx.forward(model, initial_state)

            # Initialize controller state
            controller_state = controller.init_state(ctrl_rng)

            def step_fn(carry, _unused):
                """Single simulation step with controller."""
                state, ctrl_state = carry
                current_state = extract_state_features(state)
                current_accel = state.qacc

                # Get action from controller (returns action and new state)
                action, new_ctrl_state = controller.get_action(
                    state, ctrl_state
                )

                # Step simulation
                state = state.replace(ctrl=action)
                next_state = mjx.step(model, state)
                next_state_features = extract_state_features(next_state)

                return (next_state, new_ctrl_state), (
                    current_state,
                    action,
                    next_state_features,
                    current_accel,
                )

            _, transitions = jax.lax.scan(
                step_fn,
                (initial_state, controller_state),
                jnp.arange(rollout_length),
            )
            return transitions

        jit_single_rollout = jax.jit(single_rollout)
        rng_keys = jax.random.split(rng, num_rollouts)
        all_transitions = jax.vmap(jit_single_rollout)(rng_keys)

    # Unpack and reshape: (num_rollouts, rollout_length, ...) -> (N, ...)
    states = all_transitions[0].reshape(-1, model.nq + model.nv)
    actions = all_transitions[1].reshape(-1, model.nu)
    next_states = all_transitions[2].reshape(-1, model.nq + model.nv)
    accelerations = all_transitions[3].reshape(-1, model.nv)

    return states, actions, next_states, accelerations


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

    This is a convenience wrapper around collect_rollouts with controller=None.

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
    return collect_rollouts(  # noqa: E501
        model=model,
        num_rollouts=num_rollouts,
        rollout_length=rollout_length,
        action_min=action_min,
        action_max=action_max,
        rng=rng,
        controller=None,
        initial_state_sampler=initial_state_sampler,
    )


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


def save_dataset(dataset: DynamicsDataset, path: str | Path) -> None:
    """Save a DynamicsDataset to disk.

    Args:
        dataset: Dataset to save.
        path: Path to save the dataset (uses .pkl extension if not provided).
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".pkl")

    # Convert JAX arrays to numpy for serialization
    data_dict = {
        "states": np.array(dataset.states),
        "actions": np.array(dataset.actions),
        "next_states": np.array(dataset.next_states),
        "accelerations": np.array(dataset.accelerations),
        "state_dim": dataset.state_dim,
        "action_dim": dataset.action_dim,
        "nq": dataset.nq,
        "nv": dataset.nv,
        "dt": dataset.dt,
    }

    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data_dict, f)


def load_dataset(path: str | Path) -> DynamicsDataset:
    """Load a DynamicsDataset from disk.

    Args:
        path: Path to the saved dataset.

    Returns:
        Loaded DynamicsDataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "rb") as f:
        data_dict = pickle.load(f)

    # Convert numpy arrays back to JAX arrays
    return DynamicsDataset(
        states=jnp.array(data_dict["states"]),
        actions=jnp.array(data_dict["actions"]),
        next_states=jnp.array(data_dict["next_states"]),
        accelerations=jnp.array(data_dict["accelerations"]),
        state_dim=data_dict["state_dim"],
        action_dim=data_dict["action_dim"],
        nq=data_dict["nq"],
        nv=data_dict["nv"],
        dt=data_dict["dt"],
    )


def collect_and_prepare_data(
    env,
    num_rollouts: int = 50,
    rollout_length: int = 100,
    action_min: Optional[jax.Array] = None,
    action_max: Optional[jax.Array] = None,
    train_ratio: float = 0.8,
    rng: Optional[jax.Array] = None,
    dataset_path: Optional[str | Path] = None,
    force_recollect: bool = False,
    controller: Optional[Controller] = None,
    num_controlled_rollouts: int = 0,
) -> Tuple[DynamicsDataset, DynamicsDataset]:
    """Collect rollouts and prepare train/val datasets.

    Supports mixing random and controlled rollouts. Optionally saves/loads
    datasets from disk to avoid re-collection.

    Args:
        env: Environment instance (provides model, action bounds, reset).
        num_rollouts: Total number of rollouts to collect.
        rollout_length: Length of each rollout.
        action_min: Minimum action values (defaults to env.action_min).
        action_max: Maximum action values (defaults to env.action_max).
        train_ratio: Fraction of data for training.
        rng: Random number generator key.
        dataset_path: Optional path to save/load dataset. If provided and
            file exists, loads from disk instead of collecting.
            Use .pkl extension or omit.
        force_recollect: If True, recollect data even if dataset_path exists.
        controller: Optional controller for controlled rollouts.
        num_controlled_rollouts: Number of controlled rollouts to collect.
            Remaining rollouts will be random. If controller is None and
            num_controlled_rollouts > 0, raises ValueError.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    # Try to load from disk if path provided and exists
    if dataset_path is not None and not force_recollect:
        dataset_path = Path(dataset_path)
        if not dataset_path.suffix:
            dataset_path = dataset_path.with_suffix(".pkl")

        if dataset_path.exists():
            print(f"Loading dataset from {dataset_path}...")
            full_dataset = load_dataset(dataset_path)

            # Split into train/val
            if rng is None:
                rng = jax.random.PRNGKey(0)
            rng, split_rng = jax.random.split(rng)
            train_dataset, val_dataset = split_dataset(
                full_dataset, train_ratio=train_ratio, rng=split_rng
            )
            print(f"Loaded dataset: {len(full_dataset)} samples")
            return train_dataset, val_dataset

    # Validate controller arguments
    num_random_rollouts = num_rollouts - num_controlled_rollouts
    if num_controlled_rollouts > 0 and controller is None:
        raise ValueError(
            "controller must be provided when num_controlled_rollouts > 0"
        )
    if num_random_rollouts < 0:
        raise ValueError(
            "num_controlled_rollouts cannot exceed num_rollouts"
        )

    # Collect new data
    if rng is None:
        rng = jax.random.PRNGKey(0)

    model = env.model
    if action_min is None:
        action_min = env.action_min
    if action_max is None:
        action_max = env.action_max

    # Use env's reset method
    def initial_state_sampler(m, r):
        data = mjx.make_data(m)
        data = env.reset(data, r)
        return mjx.forward(m, data)

    # Collect random rollouts
    all_states = []
    all_actions = []
    all_next_states = []
    all_accelerations = []

    if num_random_rollouts > 0:
        print(
            f"Collecting {num_random_rollouts} random rollouts of length "
            f"{rollout_length}..."
        )
        rng, subrng = jax.random.split(rng)
        states, actions, next_states, accelerations = collect_rollouts(
            model=model,
            num_rollouts=num_random_rollouts,
            rollout_length=rollout_length,
            action_min=action_min,
            action_max=action_max,
            rng=subrng,
            controller=None,
            initial_state_sampler=initial_state_sampler,
        )
        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        all_accelerations.append(accelerations)

    # Collect controlled rollouts
    if num_controlled_rollouts > 0:
        print(
            f"Collecting {num_controlled_rollouts} controlled rollouts "
            f"of length {rollout_length}..."
        )
        rng, subrng = jax.random.split(rng)
        states, actions, next_states, accelerations = collect_rollouts(
            model=model,
            num_rollouts=num_controlled_rollouts,
            rollout_length=rollout_length,
            action_min=action_min,
            action_max=action_max,
            rng=subrng,
            controller=controller,
            initial_state_sampler=initial_state_sampler,
        )
        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        all_accelerations.append(accelerations)

    # Concatenate all rollouts
    states = (
        jnp.concatenate(all_states, axis=0) if all_states else jnp.array([])
    )
    actions = (
        jnp.concatenate(all_actions, axis=0) if all_actions else jnp.array([])
    )
    next_states = (
        jnp.concatenate(all_next_states, axis=0)
        if all_next_states
        else jnp.array([])
    )
    accelerations = (
        jnp.concatenate(all_accelerations, axis=0)
        if all_accelerations
        else jnp.array([])
    )

    dataset = create_dataset(
        model=model,
        states=states,
        actions=actions,
        next_states=next_states,
        accelerations=accelerations,
        dt=env.dt,
    )

    # Save to disk if path provided
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        if not dataset_path.suffix:
            dataset_path = dataset_path.with_suffix(".pkl")
        print(f"Saving dataset to {dataset_path}...")
        save_dataset(dataset, dataset_path)

    rng, split_rng = jax.random.split(rng)
    train_dataset, val_dataset = split_dataset(
        dataset, train_ratio=train_ratio, rng=split_rng
    )

    return train_dataset, val_dataset

