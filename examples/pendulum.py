"""Train and test a neural dynamics model for pendulum with MPC."""

import argparse
from pathlib import Path

import jax
import mujoco
import numpy as np

from dynax import TrainingConfig, train_dynamics_model
from dynax.architectures import ResNetNeuralModel
from dynax.envs import PendulumEnv
from dynax.mpc import NeuralTask
from dynax.utils import HydraxController, collect_and_prepare_data
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pendulum import Pendulum as HydraxPendulumTask


def train_model(model_path: str = "models/pendulum_neural.pkl"):
    """Train a neural dynamics model and save it to disk.

    Args:
        model_path: Path to save the trained model.
    """
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)

    # Create hydrax task and controller for data collection
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
    print("\nCollecting training data...")
    env = PendulumEnv()
    rng = jax.random.PRNGKey(0)
    train_dataset, val_dataset = collect_and_prepare_data(
        env=env,
        num_rollouts=500,
        rollout_length=100,
        rng=rng,
        dataset_path="data/pendulum_dataset.pkl",
        controller=controller,
        num_controlled_rollouts=250,
        force_recollect=False,
    )

    # Create model with history_length=1 for NeuralTask compatibility
    print("\nCreating model...")
    dynamics_model = ResNetNeuralModel(
        env=env,
        history_length=1,
    )

    # Train model
    print("\nTraining model...")
    rng, train_rng = jax.random.split(rng)
    trained_params = train_dynamics_model(
        model=dynamics_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=TrainingConfig(
            num_epochs=200,
            batch_size=512,
            learning_rate=1e-3,
            noise_std=0.0,
            log_dir="logs/pendulum",
            render_videos=False,
        ),
        rng=train_rng,
        env=env,
        eval_num_rollouts=5,
    )

    # Save model
    print(f"\nSaving model to {model_path}...")
    dynamics_model.save_model(trained_params, model_path)

    print(f"Model saved successfully to {model_path}")
    print("=" * 60)


def mpc_model(model_path: str = "models/pendulum_neural.pkl"):
    """Load a trained model and run interactive MPC simulation.

    Args:
        model_path: Path to the saved model.
    """
    print("=" * 60)
    print("MPC MODE - Interactive MPC Simulation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create environment and model architecture (must match training)
    env = PendulumEnv()
    dynamics_model = ResNetNeuralModel(
        env=env,
        history_length=1,
    )

    # Load model parameters
    print("Loading model parameters...")
    model_params = dynamics_model.load_model(model_path)

    # Create base task and neural task
    print("Creating neural task...")
    base_task = HydraxPendulumTask()
    neural_task = NeuralTask(
        base_task=base_task,
        dynamics_model=dynamics_model,
        model_params=model_params,
    )

    # Create MPC controller using the neural task
    print("Creating Predictive Sampling controller...")
    ctrl = PredictiveSampling(
        neural_task,
        num_samples=32,
        noise_level=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Set up simulation
    mj_model = neural_task.mj_model
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = np.array([0.0])
    mj_data.qvel[:] = np.array([0.0])

    # Run interactive simulation
    print("\nStarting interactive simulation...")
    print("Close the viewer window to exit.")
    print("=" * 60)
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=50.0,
        fixed_camera_id=0,
        show_traces=False,
        max_traces=1,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a neural dynamics model or run MPC for pendulum."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train mode
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--model-path",
        type=str,
        default="models/pendulum_neural.pkl",
        help="Path to save the trained model",
    )

    # MPC mode
    mpc_parser = subparsers.add_parser(
        "mpc", help="Run MPC with the trained model"
    )
    mpc_parser.add_argument(
        "--model-path",
        type=str,
        default="models/pendulum_neural.pkl",
        help="Path to the saved model",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(model_path=args.model_path)
    elif args.mode == "mpc":
        mpc_model(model_path=args.model_path)


if __name__ == "__main__":
    main()
