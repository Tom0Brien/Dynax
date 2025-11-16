"""Train a neural dynamics model for pendulum."""

import jax
from hydrax.algs import PredictiveSampling
from hydrax.tasks.pendulum import Pendulum as HydraxPendulumTask

from dynax import TrainingConfig, train_dynamics_model
from dynax.architectures import ResNetDynamicsModel, TransformerDynamicsModel
from dynax.envs import PendulumEnv
from dynax.utils import HydraxController, collect_and_prepare_data

# Create hydrax task and controller
hydrax_task = HydraxPendulumTask()
hydrax_controller = PredictiveSampling(
    hydrax_task,
    num_samples=32,
    noise_level=0.1,
    plan_horizon=1.0,
    spline_type="zero",
    num_knots=11,
)
controller = HydraxController(hydrax_controller)

# Create environment and collect data
env = PendulumEnv()
rng = jax.random.PRNGKey(0)
train_dataset, val_dataset = collect_and_prepare_data(
    env=env,
    num_rollouts=50,
    rollout_length=100,
    rng=rng,
    dataset_path="data/pendulum_dataset.pkl",
    controller=controller,
    num_controlled_rollouts=20,  # 20 controlled + 30 random
)

# Train model
dynamics_model = TransformerDynamicsModel(
    env=env,
    history_length=10,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    ff_dim=1024,
    dropout=0.1,
    activation="gelu",
)

rng, train_rng = jax.random.split(rng)
trained_params = train_dynamics_model(
    model=dynamics_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=TrainingConfig(
        num_epochs=1000,
        batch_size=512,
        learning_rate=1e-3,
        noise_std=0.01,
        log_dir="logs/pendulum",  # TensorBoard logging
        render_videos=False,
    ),
    rng=train_rng,
    env=env,
    eval_num_rollouts=5,
)
