"""Train a neural dynamics model for pendulum."""

import jax

from dynax import TrainingConfig, train_dynamics_model
from dynax.architectures import MLPDynamicsModel
from dynax.envs import PendulumEnv
from dynax.utils import collect_and_prepare_data

# Create environment and collect data
env = PendulumEnv()
rng = jax.random.PRNGKey(0)
train_dataset, val_dataset = collect_and_prepare_data(
    env=env,
    num_rollouts=50,
    rollout_length=100,
    rng=rng,
)

# Train model (using paper architecture)
dynamics_model = MLPDynamicsModel(
    state_dim=train_dataset.state_dim,
    action_dim=train_dataset.action_dim,
    hidden_dims=(500, 500),
    activation="relu",
)

rng, train_rng = jax.random.split(rng)
trained_params = train_dynamics_model(
    model=dynamics_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=TrainingConfig(
        num_epochs=100,
        batch_size=512,
        learning_rate=1e-3,
        noise_std=0.01,
    ),
    rng=train_rng,
    env=env,
)
