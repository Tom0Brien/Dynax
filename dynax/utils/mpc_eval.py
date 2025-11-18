"""Generic utilities for MPC evaluation and comparison."""

from pathlib import Path
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mujoco import mjx

from dynax.envs.envs import Env
from dynax.utils.data import extract_state_features
from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import Task


def create_state_sync_fn(
    target_model: mjx.Model,
) -> Callable[[mjx.Data], mjx.Data]:
    """Create a function to sync state from one model to another.

    Args:
        target_model: The target MJX model to sync state to.

    Returns:
        A function that takes a source state and returns a synced state.
    """
    @jax.jit
    def sync_state(source_state: mjx.Data) -> mjx.Data:
        """Sync state from source model to target model."""
        target_data = mjx.make_data(target_model)
        # Copy qpos, qvel, time, and ctrl from source state
        target_data = target_data.replace(
            qpos=source_state.qpos,
            qvel=source_state.qvel,
            time=source_state.time,
            ctrl=source_state.ctrl,
        )
        # Copy mocap data if present
        if source_state.mocap_pos.shape[0] > 0:
            target_data = target_data.replace(
                mocap_pos=source_state.mocap_pos,
                mocap_quat=source_state.mocap_quat,
            )
        # Forward kinematics to update dependent quantities
        return mjx.forward(target_model, target_data)

    return sync_state


def run_mpc_episode(
    initial_state: mjx.Data,
    controller: SamplingBasedController,
    true_step_fn: Callable[[mjx.Data, jax.Array], mjx.Data],
    sync_fn: Callable[[mjx.Data], mjx.Data],
    cost_fn: Callable[[mjx.Data, jax.Array], jax.Array],
    terminal_cost_fn: Callable[[mjx.Data], jax.Array],
    episode_length: int,
    dt: float,
    replan_freq_hz: float = 10.0,
    return_trajectory: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Run a single MPC episode with replanning.

    Args:
        initial_state: Initial state for the episode.
        controller: MPC controller to use.
        true_step_fn: Function to step true dynamics:
            (state, action) -> next_state.
        sync_fn: Function to sync state from true model to controller model.
        cost_fn: Running cost function: (state, action) -> cost.
        terminal_cost_fn: Terminal cost function: (state) -> cost.
        episode_length: Number of steps in the episode.
        dt: Time step duration.
        replan_freq_hz: Replanning frequency in Hz.
        return_trajectory: Whether to return trajectory.

    Returns:
        Tuple of (total_cost, trajectory) if return_trajectory=True,
        else (total_cost, empty_array).
    """
    true_state = initial_state
    total_cost = 0.0

    # Initialize controller with synced state
    ctrl_state = sync_fn(true_state)
    ctrl_params = controller.init_params(seed=0)

    # JIT-compiled optimize and interpolation functions
    jit_optimize = jax.jit(controller.optimize)
    jit_interp = jax.jit(controller.interp_func)

    # Replanning frequency
    replan_period = 1.0 / replan_freq_hz
    sim_steps_per_replan = max(1, int(replan_period / dt))

    def step_fn(carry, step_idx):
        """Single MPC step."""
        true_state, ctrl_state, ctrl_params, total_cost = carry

        # Sync controller state from true state
        ctrl_state = sync_fn(true_state)

        # Replan if needed
        should_replan = step_idx % sim_steps_per_replan == 0

        def replan(s, p):
            """Replan."""
            new_params, _ = jit_optimize(s, p)
            return new_params

        def no_replan(s, p):
            """Don't replan."""
            return p

        # Conditionally replan using synced controller state
        ctrl_params = jax.lax.cond(
            should_replan,
            replan,
            no_replan,
            ctrl_state,
            ctrl_params,
        )

        # Get action from current policy
        t_curr = ctrl_state.time
        tq = jnp.array([t_curr])
        tk = ctrl_params.tk
        knots = ctrl_params.mean[None, ...]  # (1, num_knots, nu)
        actions = jit_interp(tq, tk, knots)
        action = actions[0, 0]  # (nu,)

        # Clip action to limits
        action = jnp.clip(action, controller.task.u_min, controller.task.u_max)

        # Compute cost using synced state (for consistency)
        cost = cost_fn(ctrl_state, action)
        total_cost = total_cost + cost

        # Step true dynamics (this is what actually happens)
        next_true_state = true_step_fn(true_state, action)

        # Extract state features for trajectory tracking (after step)
        state_features = extract_state_features(next_true_state)

        return (
            (next_true_state, ctrl_state, ctrl_params, total_cost),
            state_features,
        )

    # Run episode
    (final_true_state, _, _, total_cost), trajectory = jax.lax.scan(
        step_fn,
        (true_state, ctrl_state, ctrl_params, total_cost),
        jnp.arange(episode_length),
    )

    # Add initial state to trajectory
    initial_features = extract_state_features(initial_state)
    trajectory = jnp.concatenate(
        [initial_features[None, :], trajectory], axis=0
    )

    # Add terminal cost using synced final state
    final_ctrl_state = sync_fn(final_true_state)
    terminal_cost = terminal_cost_fn(final_ctrl_state)
    total_cost = total_cost + terminal_cost

    if return_trajectory:
        return total_cost, trajectory
    return total_cost, jnp.array([])


def create_true_step_fn(
    true_env: Env,
) -> Callable[[mjx.Data, jax.Array], mjx.Data]:
    """Create a JIT-compiled step function for true dynamics.

    Args:
        true_env: Environment with true dynamics.

    Returns:
        JIT-compiled step function: (state, action) -> next_state.
    """
    @jax.jit
    def true_step(data: mjx.Data, action: jax.Array) -> mjx.Data:
        """Step true dynamics."""
        data = data.replace(ctrl=action)
        return mjx.step(true_env.model, data)

    return true_step


def compute_mpc_statistics(
    costs: np.ndarray,
) -> dict:
    """Compute statistics from MPC episode costs.

    Args:
        costs: Array of episode costs, shape (num_episodes,).

    Returns:
        Dictionary with statistics: mean, std, min, max.
    """
    return {
        "mean": np.mean(costs),
        "std": np.std(costs),
        "min": np.min(costs),
        "max": np.max(costs),
    }


def print_mpc_comparison(
    base_stats: dict,
    learned_stats: dict,
    num_episodes: int,
    episode_length: int,
) -> None:
    """Print MPC performance comparison.

    Args:
        base_stats: Statistics for base controller.
        learned_stats: Statistics for learned controller.
        num_episodes: Number of episodes evaluated.
        episode_length: Length of each episode.
    """
    improvement = (
        (base_stats["mean"] - learned_stats["mean"]) / base_stats["mean"]
    ) * 100

    print("\n" + "=" * 80)
    print("MPC Performance Comparison")
    print("=" * 80)
    print(f"\nEpisodes: {num_episodes}")
    print(f"Episode length: {episode_length} steps")
    print("\nBase Physics MPC:")
    print(f"  Mean cost: {base_stats['mean']:.4f} ± {base_stats['std']:.4f}")
    print(f"  Min cost:  {base_stats['min']:.4f}")
    print(f"  Max cost:  {base_stats['max']:.4f}")

    print("\nLearned Neural MPC:")
    print(
        f"  Mean cost: {learned_stats['mean']:.4f} ± "
        f"{learned_stats['std']:.4f}"
    )
    print(f"  Min cost:  {learned_stats['min']:.4f}")
    print(f"  Max cost:  {learned_stats['max']:.4f}")
    print(f"\nImprovement: {improvement:.2f}% cost reduction")
    print("=" * 80)


def setup_mpc_evaluation(
    base_task: Task,
    neural_task: Task,
    true_env: Env,
    base_env: Env,
) -> Tuple[
    Callable[[mjx.Data], mjx.Data],
    Callable[[mjx.Data], mjx.Data],
    Callable[[mjx.Data, jax.Array], mjx.Data],
    Callable[[mjx.Data, jax.Array], jax.Array],
    Callable[[mjx.Data], jax.Array],
]:
    """Set up MPC evaluation with state syncing and cost functions.

    Args:
        base_task: Base task for MPC.
        neural_task: Neural task for MPC.
        true_env: Environment with true dynamics.
        base_env: Base environment.

    Returns:
        Tuple of (sync_to_base_fn, sync_to_neural_fn, true_step_fn,
        cost_fn, terminal_cost_fn).
    """
    # Create state sync functions
    sync_to_base_fn = create_state_sync_fn(base_env.model)
    sync_to_neural_fn = create_state_sync_fn(neural_task.model)

    # Create true step function
    true_step_fn = create_true_step_fn(true_env)

    # Create cost functions (using base task for consistency)
    @jax.jit
    def cost_fn(state: mjx.Data, action: jax.Array) -> jax.Array:
        """Compute running cost."""
        return base_task.running_cost(state, action)

    @jax.jit
    def terminal_cost_fn(state: mjx.Data) -> jax.Array:
        """Compute terminal cost."""
        return base_task.terminal_cost(state)

    return (
        sync_to_base_fn,
        sync_to_neural_fn,
        true_step_fn,
        cost_fn,
        terminal_cost_fn,
    )


def plot_mpc_comparison(
    base_costs: np.ndarray,
    learned_costs: np.ndarray,
    base_stats: dict,
    learned_stats: dict,
    plot_path: str,
    color_scheme: dict = None,
) -> None:
    """Create MPC comparison plots.

    Args:
        base_costs: Array of base controller episode costs.
        learned_costs: Array of learned controller episode costs.
        base_stats: Statistics dictionary for base controller.
        learned_stats: Statistics dictionary for learned controller.
        plot_path: Path to save the plot.
        color_scheme: Optional color scheme dictionary with keys:
            'light', 'medium', 'dark'. Defaults to light blue theme.
    """
    if color_scheme is None:
        color_scheme = {
            "light": "#87CEEB",  # Sky blue
            "medium": "#4682B4",  # Steel blue
            "dark": "#1E90FF",  # Dodger blue
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cost distribution
    axes[0].hist(
        base_costs,
        bins=15,
        alpha=0.7,
        label="Base Physics MPC",
        color=color_scheme["light"],
        edgecolor=color_scheme["medium"],
    )
    axes[0].hist(
        learned_costs,
        bins=15,
        alpha=0.7,
        label="Learned Neural MPC",
        color=color_scheme["dark"],
        edgecolor=color_scheme["medium"],
    )
    axes[0].axvline(
        base_stats["mean"],
        color=color_scheme["medium"],
        linestyle="--",
        linewidth=2,
    )
    axes[0].axvline(
        learned_stats["mean"],
        color=color_scheme["dark"],
        linestyle="--",
        linewidth=2,
    )
    axes[0].set_xlabel("Total Episode Cost")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Cost Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Comparison bar plot
    categories = ["Base Physics\nMPC", "Learned Neural\nMPC"]
    means = [base_stats["mean"], learned_stats["mean"]]
    stds = [base_stats["std"], learned_stats["std"]]
    colors = [color_scheme["light"], color_scheme["dark"]]

    bars = axes[1].bar(
        categories,
        means,
        yerr=stds,
        capsize=10,
        color=colors,
        edgecolor=color_scheme["medium"],
    )
    axes[1].set_ylabel("Mean Episode Cost")
    axes[1].set_title("MPC Performance Comparison")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def _evaluate_mpc_controllers_core(
    base_controller: SamplingBasedController,
    learned_controller: SamplingBasedController,
    initial_states: mjx.Data,
    true_step_fn: Callable[[mjx.Data, jax.Array], mjx.Data],
    sync_to_base_fn: Callable[[mjx.Data], mjx.Data],
    sync_to_neural_fn: Callable[[mjx.Data], mjx.Data],
    cost_fn: Callable[[mjx.Data, jax.Array], jax.Array],
    terminal_cost_fn: Callable[[mjx.Data], jax.Array],
    episode_length: int,
    dt: float,
    rng: jax.Array,
    replan_freq_hz: float = 10.0,
) -> Tuple[jax.Array, jax.Array]:
    """Evaluate base and learned MPC controllers in parallel (low-level).

    Args:
        base_controller: Base physics MPC controller.
        learned_controller: Learned neural MPC controller.
        initial_states: Initial states for episodes (pytree from vmap).
        true_step_fn: Function to step true dynamics.
        sync_to_base_fn: Function to sync state to base model.
        sync_to_neural_fn: Function to sync state to neural model.
        cost_fn: Running cost function.
        terminal_cost_fn: Terminal cost function.
        episode_length: Length of each episode.
        dt: Time step duration.
        rng: Random number generator key.
        replan_freq_hz: Replanning frequency in Hz.

    Returns:
        Tuple of (base_costs, learned_costs) arrays.
    """
    # Get number of episodes from pytree structure
    num_episodes = initial_states.qpos.shape[0]

    # Create episode runner functions
    def run_base_episode(state, _rng):
        return run_mpc_episode(
            state,
            base_controller,
            true_step_fn,
            sync_to_base_fn,
            cost_fn,
            terminal_cost_fn,
            episode_length,
            dt,
            replan_freq_hz=replan_freq_hz,
            return_trajectory=False,
        )[0]  # Extract cost only

    def run_learned_episode(state, _rng):
        return run_mpc_episode(
            state,
            learned_controller,
            true_step_fn,
            sync_to_neural_fn,
            cost_fn,
            terminal_cost_fn,
            episode_length,
            dt,
            replan_freq_hz=replan_freq_hz,
            return_trajectory=False,
        )[0]  # Extract cost only

    # Generate RNG keys for episodes
    rng, base_rng = jax.random.split(rng)
    rng, learned_rng = jax.random.split(rng)
    base_rngs = jax.random.split(base_rng, num_episodes)
    learned_rngs = jax.random.split(learned_rng, num_episodes)

    # Run episodes in parallel
    base_costs = jax.vmap(run_base_episode)(initial_states, base_rngs)
    learned_costs = jax.vmap(run_learned_episode)(initial_states, learned_rngs)

    return base_costs, learned_costs


def evaluate_mpc_controllers(
    base_controller: SamplingBasedController,
    learned_controller: SamplingBasedController,
    initial_states: mjx.Data,
    base_task: Task,
    neural_task: Task,
    true_env: Env,
    base_env: Env,
    episode_length: int,
    dt: float,
    rng: jax.Array,
    plot_path: str = None,
    replan_freq_hz: float = 10.0,
) -> Tuple[jax.Array, jax.Array, dict, dict]:
    """Evaluate MPC controllers and generate statistics and plots.

    This is a convenience function that combines setup, evaluation,
    statistics computation, printing, and plotting.

    Args:
        base_controller: Base physics MPC controller.
        learned_controller: Learned neural MPC controller.
        initial_states: Initial states for episodes (pytree from vmap).
        base_task: Base task for MPC.
        neural_task: Neural task for MPC.
        true_env: Environment with true dynamics.
        base_env: Base environment.
        episode_length: Length of each episode.
        dt: Time step duration.
        rng: Random number generator key.
        plot_path: Optional path to save comparison plot.
        replan_freq_hz: Replanning frequency in Hz.

    Returns:
        Tuple of (base_costs, learned_costs, base_stats, learned_stats).
    """
    # Setup MPC evaluation utilities
    (
        sync_to_base_fn,
        sync_to_neural_fn,
        true_step_fn,
        cost_fn,
        terminal_cost_fn,
    ) = setup_mpc_evaluation(base_task, neural_task, true_env, base_env)

    # Evaluate controllers
    base_costs, learned_costs = _evaluate_mpc_controllers_core(
        base_controller,
        learned_controller,
        initial_states,
        true_step_fn,
        sync_to_base_fn,
        sync_to_neural_fn,
        cost_fn,
        terminal_cost_fn,
        episode_length,
        dt,
        rng,
        replan_freq_hz=replan_freq_hz,
    )

    # Compute statistics
    base_stats = compute_mpc_statistics(np.array(base_costs))
    learned_stats = compute_mpc_statistics(np.array(learned_costs))

    # Print comparison
    num_episodes = initial_states.qpos.shape[0]
    print_mpc_comparison(
        base_stats, learned_stats, num_episodes, episode_length
    )

    # Create plot if path provided
    if plot_path is not None:
        plot_mpc_comparison(
            np.array(base_costs),
            np.array(learned_costs),
            base_stats,
            learned_stats,
            plot_path,
        )

    return base_costs, learned_costs, base_stats, learned_stats


def render_mpc_trajectories(
    base_controller: SamplingBasedController,
    learned_controller: SamplingBasedController,
    initial_states: mjx.Data,
    base_task: Task,
    neural_task: Task,
    true_env: Env,
    base_env: Env,
    episode_length: int,
    dt: float,
    rng: jax.Array,
    log_dir: str,
    num_videos: int = 1,
    replan_freq_hz: float = 10.0,
    render_fn=None,
) -> None:
    """Render trajectory videos for MPC controllers.

    Args:
        base_controller: Base physics MPC controller.
        learned_controller: Learned neural MPC controller.
        initial_states: Initial states for episodes (pytree from vmap).
        base_task: Base task for MPC.
        neural_task: Neural task for MPC.
        true_env: Environment with true dynamics.
        base_env: Base environment.
        episode_length: Length of each episode.
        dt: Time step duration.
        rng: Random number generator key.
        log_dir: Directory to save videos.
        num_videos: Number of videos to render.
        replan_freq_hz: Replanning frequency in Hz.
        render_fn: Optional rendering function.
            Defaults to render_single_trajectory_video.
    """
    if render_fn is None:
        from dynax.evaluation import render_single_trajectory_video

        render_fn = render_single_trajectory_video

    # Setup MPC evaluation utilities
    (
        sync_to_base_fn,
        sync_to_neural_fn,
        true_step_fn,
        cost_fn,
        terminal_cost_fn,
    ) = setup_mpc_evaluation(base_task, neural_task, true_env, base_env)

    num_episodes = initial_states.qpos.shape[0]
    num_videos = min(num_videos, num_episodes)

    rng, _ = jax.random.split(rng)

    for vid_idx in range(num_videos):
        # Run base controller episode with trajectory
        _, base_traj = run_mpc_episode(
            initial_states[vid_idx],
            base_controller,
            true_step_fn,
            sync_to_base_fn,
            cost_fn,
            terminal_cost_fn,
            episode_length,
            dt,
            replan_freq_hz=replan_freq_hz,
            return_trajectory=True,
        )

        # Run learned controller episode with trajectory
        _, learned_traj = run_mpc_episode(
            initial_states[vid_idx],
            learned_controller,
            true_step_fn,
            sync_to_neural_fn,
            cost_fn,
            terminal_cost_fn,
            episode_length,
            dt,
            replan_freq_hz=replan_freq_hz,
            return_trajectory=True,
        )

        # Render videos
        base_video_path = f"{log_dir}/base_mpc_episode_{vid_idx}.mp4"
        print(f"  Rendering base MPC video: {base_video_path}")
        render_fn(
            env=true_env,
            states=base_traj,
            fps=30,
            save_path=base_video_path,
            max_frames=episode_length,
        )

        learned_video_path = f"{log_dir}/learned_mpc_episode_{vid_idx}.mp4"
        print(f"  Rendering learned MPC video: {learned_video_path}")
        render_fn(
            env=true_env,
            states=learned_traj,
            fps=30,
            save_path=learned_video_path,
            max_frames=episode_length,
        )

