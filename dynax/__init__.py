"""Dynax: Training neural network models for robotic systems with MuJoCo MJX."""

import os
from pathlib import Path

import jax

# Package root
ROOT = str(Path(__file__).parent.absolute())

# Set XLA flags for better performance
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

# Import core functionality
from dynax.base import BaseDynamicsModel, DynamicsModelParams
from dynax.models import (
    list_available_models,
    load_mjx_model,
    load_model,
)
from dynax.training import (
    TrainingConfig,
    TrainingState,
    create_epoch_train_fn,
    prepare_epoch_batches,
    train_dynamics_model,
)

__version__ = "0.1.0"
__all__ = [
    "ROOT",
    "BaseDynamicsModel",
    "DynamicsModelParams",
    "TrainingConfig",
    "TrainingState",
    "train_dynamics_model",
    "create_epoch_train_fn",
    "prepare_epoch_batches",
    "load_model",
    "load_mjx_model",
    "list_available_models",
]

