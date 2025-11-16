When I ask you to write code, I only want you to write the code. Do not create any documentation, readmes, or tests unless I explicitly ask you to.

* **Prefer pure, functional code.**
  Avoid side effects, global state, in-place mutation, and data-dependent control flow that can’t be traced; structure code as pure functions over arrays.

* **Batch work aggressively.**
  Design APIs to accept and process batches (extra leading dimensions) so `vmap` can be used instead of Python loops.

* **Fuse computations with `jit`.**
  Wrap performance-critical functions with `jax.jit`, and aim for a small number of large `jit`ted functions instead of many tiny ones.

* **Keep shapes and dtypes stable.**
  Avoid changing array shapes or dtypes across calls to a `jit`ted function; prefer static shapes and consistent dtypes to maximize XLA reuse.

* **Use JAX arrays and PRNGs everywhere.**
  Use `jax.numpy` instead of NumPy, and `jax.random` with explicit PRNG keys instead of global RNGs.

* **Minimize Python control flow in hot paths.**
  Replace major Python loops/conditionals with JAX primitives (`lax.scan`, `lax.cond`, `lax.while_loop`) when they’re inside `jit`ted regions.

* **Exploit parallel transforms.**
  Prefer `vmap` for data parallelism on one device; consider `pmap`/`pjit` for multi-device parallelism when relevant.

* **Avoid host–device thrashing.**
  Don’t repeatedly move data between CPU and accelerator; keep large arrays on device and avoid frequent `.block_until_ready()` or `.item()` calls in tight loops.

* **Use JAX-native libraries.**
  Prefer JAX-based ecosystem tools (e.g., Optax/Flax/etc. when appropriate) over rolling custom Python that re-implements similar logic.

