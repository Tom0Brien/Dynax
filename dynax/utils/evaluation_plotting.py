import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mujoco import mjx
from pathlib import Path
from typing import List, Optional, Tuple

from dynax.evaluation import (
    create_rollout_fn,
    create_true_rollout_fn,
)
from dynax.utils.data import extract_state_features

def evaluate_and_plot_model_comparison(
    dynamics_model,
    trained_params,
    base_env,
    varied_env,
    rng,
    num_test_rollouts: int = 10,
    rollout_length: int = 100,
    plot_path: str = "comparison_plot.png",
    state_indices_to_plot: List[int] = [0, 1],
    state_labels: List[str] = ["State 0", "State 1"],
):
    """
    Evaluates the learned model against the base and true physics models,
    computes statistics, and generates comparison plots.

    Args:
        dynamics_model: The learned ResidualNeuralModel.
        trained_params: The trained parameters.
        base_env: The base environment (for base physics baseline).
        varied_env: The varied environment (ground truth).
        rng: JAX random key.
        num_test_rollouts: Number of rollouts (default 10).
        rollout_length: Length of each rollout (default 100).
        plot_path: Path to save the comparison plot.
        state_indices_to_plot: List of state indices to plot in the sample trajectory.
        state_labels: List of labels for the state indices.
    """
    print("\n" + "=" * 80)
    print("Comparing Learned Residual Model vs Base Physics Model")
    print("=" * 80)

    # Create rollout functions
    learned_rollout_fn = create_rollout_fn(dynamics_model, trained_params)
    base_physics_rollout_fn = create_true_rollout_fn(base_env)
    true_physics_rollout_fn = create_true_rollout_fn(varied_env)

    rng, test_rng = jax.random.split(rng)
    test_rngs = jax.random.split(test_rng, num_test_rollouts)

    # Create initial states
    def create_initial_state_history(rng_key, env):
        """Create initial state history for rollout."""
        data = mjx.make_data(env.model)
        data = env.reset(data, rng_key)
        data = mjx.forward(env.model, data)
        initial_state = extract_state_features(data)
        # Repeat for history length
        history_length = dynamics_model.history_length
        return jnp.repeat(initial_state[None, :], history_length, axis=0), data

    # Generate random actions
    rng, action_rng = jax.random.split(rng)
    # Handle cases where action limits might be scalar or array
    action_min = varied_env.action_min
    action_max = varied_env.action_max
    
    actions_batch = jax.random.uniform(
        action_rng,
        (num_test_rollouts, rollout_length, varied_env.model.nu),
        minval=action_min,
        maxval=action_max,
    )

    # Run rollouts
    all_true_trajs = []
    all_base_trajs = []
    all_learned_trajs = []

    for i in range(num_test_rollouts):
        # Create initial state history and data
        (
            initial_state_history,
            initial_varied_data,
        ) = create_initial_state_history(test_rngs[i], varied_env)

        # True physics (varied model) - use the data directly from reset
        true_data = initial_varied_data

        # Base physics (base model) - copy qpos, qvel, and mocap data
        base_data = mjx.make_data(base_env.model)
        
        # Prepare replacement dictionary
        replace_kwargs = {
            "qpos": true_data.qpos,
            "qvel": true_data.qvel,
        }
        
        # Conditionally add mocap fields if they exist and match
        # Note: mjx.Data always has mocap_pos/quat, but size depends on model.
        # We assume base_env and varied_env have compatible structures (same number of bodies/mocaps)
        # or at least that we want to copy what we can.
        # However, mjx.replace expects arrays of correct shape.
        # If base_env has no mocap, base_data.mocap_pos is (0, 3).
        # If varied_env has no mocap, true_data.mocap_pos is (0, 3).
        # So direct assignment should work if models are compatible.
        replace_kwargs["mocap_pos"] = true_data.mocap_pos
        replace_kwargs["mocap_quat"] = true_data.mocap_quat

        base_data = base_data.replace(**replace_kwargs)
        base_data = mjx.forward(base_env.model, base_data)

        # Run rollouts
        true_traj = true_physics_rollout_fn(true_data, actions_batch[i])
        base_traj = base_physics_rollout_fn(base_data, actions_batch[i])
        learned_traj = learned_rollout_fn(
            initial_state_history, actions_batch[i]
        )

        all_true_trajs.append(true_traj)
        all_base_trajs.append(base_traj)
        all_learned_trajs.append(learned_traj)

    # Convert to numpy arrays
    true_trajs = np.array(all_true_trajs)
    base_trajs = np.array(all_base_trajs)
    learned_trajs = np.array(all_learned_trajs)

    # Compute statistics
    base_errors = np.abs(base_trajs - true_trajs)
    learned_errors = np.abs(learned_trajs - true_trajs)

    base_mae_over_time = np.mean(base_errors, axis=(0, 2))
    learned_mae_over_time = np.mean(learned_errors, axis=(0, 2))

    base_rmse_over_time = np.sqrt(
        np.mean(np.square(base_trajs - true_trajs), axis=(0, 2))
    )
    learned_rmse_over_time = np.sqrt(
        np.mean(np.square(learned_trajs - true_trajs), axis=(0, 2))
    )

    base_final_mae = np.mean(base_errors[:, -1, :])
    learned_final_mae = np.mean(learned_errors[:, -1, :])

    base_final_rmse = np.sqrt(
        np.mean(np.square(base_trajs[:, -1, :] - true_trajs[:, -1, :]))
    )
    learned_final_rmse = np.sqrt(
        np.mean(np.square(learned_trajs[:, -1, :] - true_trajs[:, -1, :]))
    )

    print(f"\nStatistics (averaged over {num_test_rollouts} rollouts):")
    print("\nBase Physics Model (with modeling error):")
    print(f"  Final MAE:  {base_final_mae:.6f}")
    print(f"  Final RMSE: {base_final_rmse:.6f}")
    print(f"  Mean MAE over time: {np.mean(base_mae_over_time):.6f}")
    print(f"  Mean RMSE over time: {np.mean(base_rmse_over_time):.6f}")

    print("\nLearned Residual Model:")
    print(f"  Final MAE:  {learned_final_mae:.6f}")
    print(f"  Final RMSE: {learned_final_rmse:.6f}")
    print(f"  Mean MAE over time: {np.mean(learned_mae_over_time):.6f}")
    print(f"  Mean RMSE over time: {np.mean(learned_rmse_over_time):.6f}")

    print("\nImprovement:")
    mae_reduction = (1 - learned_final_mae / base_final_mae) * 100
    rmse_reduction = (1 - learned_final_rmse / base_final_rmse) * 100
    print(f"  MAE reduction:  {mae_reduction:.2f}%")
    print(f"  RMSE reduction: {rmse_reduction:.2f}%")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_steps = np.arange(rollout_length + 1) * varied_env.dt

    # Plot 1: MAE over time
    axes[0, 0].plot(
        time_steps, base_mae_over_time, label="Base Physics", linewidth=2
    )
    axes[0, 0].plot(
        time_steps,
        learned_mae_over_time,
        label="Learned Residual",
        linewidth=2,
    )
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Mean Absolute Error")
    axes[0, 0].set_title("MAE Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: RMSE over time
    axes[0, 1].plot(
        time_steps, base_rmse_over_time, label="Base Physics", linewidth=2
    )
    axes[0, 1].plot(
        time_steps,
        learned_rmse_over_time,
        label="Learned Residual",
        linewidth=2,
    )
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Root Mean Squared Error")
    axes[0, 1].set_title("RMSE Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3 & 4: Sample trajectory
    traj_idx = 0
    
    for i, (state_idx, label) in enumerate(zip(state_indices_to_plot, state_labels)):
        if i >= 2: break # Only plot first 2 requested states
        
        ax = axes[1, i]
        ax.plot(
            time_steps,
            true_trajs[traj_idx, :, state_idx],
            label="True",
            linewidth=2,
            linestyle="--",
        )
        ax.plot(
            time_steps,
            base_trajs[traj_idx, :, state_idx],
            label="Base Physics",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.plot(
            time_steps,
            learned_trajs[traj_idx, :, state_idx],
            label="Learned Residual",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title(
            f"Sample Trajectory: {label} (Rollout {traj_idx+1})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    print("=" * 60)
