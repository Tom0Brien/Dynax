"""Train a neural dynamics model for cart pole."""

import jax
from hydrax.algs import PredictiveSampling
from hydrax.tasks.cart_pole import CartPole as HydraxCartPoleTask

from dynax import TrainingConfig, train_dynamics_model
from dynax.architectures import ResNetDynamicsModel, TransformerDynamicsModel
from dynax.envs import CartPoleEnv
from dynax.utils import HydraxController, collect_and_prepare_data

# Create hydrax task and controller
hydrax_task = HydraxCartPoleTask()
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
env = CartPoleEnv()
rng = jax.random.PRNGKey(0)
train_dataset, val_dataset = collect_and_prepare_data(
    env=env,
    num_rollouts=50,
    rollout_length=100,
    rng=rng,
    dataset_path="data/cart_pole_dataset.pkl",
    controller=controller,
    num_controlled_rollouts=25,  # 25 controlled + 25 random
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
        log_dir="logs/cart_pole",
        render_videos=False,
    ),
    rng=train_rng,
    env=env,
    eval_num_rollouts=5,
)

