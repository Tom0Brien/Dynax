"""Train a neural dynamics model for double cart pole."""

import jax
from hydrax.algs import PredictiveSampling
from hydrax.tasks.double_cart_pole import DoubleCartPole as HydraxDoubleCartPoleTask

from dynax import TrainingConfig, train_dynamics_model
from dynax.architectures import TransformerDynamicsModel, ResNetDynamicsModel
from dynax.envs import DoubleCartPoleEnv
from dynax.utils import HydraxController, collect_and_prepare_data

# Create hydrax task and controller
hydrax_task = HydraxDoubleCartPoleTask()
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
env = DoubleCartPoleEnv()
rng = jax.random.PRNGKey(0)
train_dataset, val_dataset = collect_and_prepare_data(
    env=env,
    num_rollouts=500,
    rollout_length=100,
    rng=rng,
    dataset_path="data/double_cart_pole_dataset.pkl",
    controller=controller,
    num_controlled_rollouts=250,  # 250 controlled + 250 random
    force_recollect=False,
)

# Train model
dynamics_model = ResNetDynamicsModel(
    env=env,
)

rng, train_rng = jax.random.split(rng)
trained_params = train_dynamics_model(
    model=dynamics_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=TrainingConfig(
        num_epochs=200,
        batch_size=512,
        learning_rate=1e-3,
        noise_std=0.01,
        log_dir="logs/double_cart_pole",
        render_videos=False,
    ),
    rng=train_rng,
    env=env,
    eval_num_rollouts=5,
)

