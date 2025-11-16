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

__version__ = "0.1.0"
__all__ = ["ROOT"]

