"""GPT2-style transformer architecture for dynamics modeling."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from dynax.base import BaseNeuralModel


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (GPT-2 style).

    Attributes:
        embed_dim: Embedding dimension.
        use_bias: Whether to use bias (default: False, GPT-2 style).
    """

    embed_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply layer normalization."""
        return nn.LayerNorm(
            epsilon=1e-5,
            use_bias=self.use_bias,
            use_scale=True,
            name="layer_norm",
        )(x)


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer (GPT2 style).

    Attributes:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate (0.0 to disable).
        use_bias: Whether to use bias in linear layers (default: False).
    """

    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        """Apply causal self-attention.

        Args:
            x: Input sequence, shape (seq_len, embed_dim).
            training: Whether in training mode (for dropout).

        Returns:
            Output sequence, shape (seq_len, embed_dim).
        """
        seq_len = x.shape[0]
        head_dim = self.embed_dim // self.num_heads

        # Compute Q, K, V (GPT-2 style: single linear projection)
        qkv = nn.Dense(
            self.embed_dim * 3,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="c_attn",
        )(x)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # q, k, v: (seq_len, num_heads, head_dim)

        # Scaled dot-product attention per head
        scale = 1.0 / jnp.sqrt(head_dim)
        # Compute attention scores per head: (num_heads, seq_len, seq_len)
        attn_weights = jnp.einsum("thd,shd->hts", q, k) * scale

        # Causal mask: only attend to past and current timesteps
        # Boolean mask: (seq_len, seq_len)
        mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1)
        # Broadcast mask over heads: (1, seq_len, seq_len)
        attn_weights = jnp.where(
            mask[None, :, :], -1e9, attn_weights
        )  # Use -1e9 instead of -inf for dtype safety
        # (num_heads, seq_len, seq_len)
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Apply dropout
        if self.dropout > 0.0 and training:
            attn_weights = nn.Dropout(self.dropout)(
                attn_weights, deterministic=not training
            )

        # Apply attention to values per head: (num_heads, seq_len, head_dim)
        out = jnp.einsum("hts,shd->thd", attn_weights, v)
        # Reshape to (seq_len, embed_dim)
        out = out.reshape(seq_len, self.embed_dim)

        # Output projection (GPT-2 style: c_proj)
        out = nn.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="c_proj",
        )(out)
        out = nn.Dropout(self.dropout, deterministic=not training)(out)

        return out


class MLP(nn.Module):
    """Feedforward MLP (GPT-2 style).

    Attributes:
        embed_dim: Embedding dimension.
        ff_dim: Feedforward hidden dimension (typically 4 * embed_dim).
        dropout: Dropout rate.
        use_bias: Whether to use bias in linear layers.
        activation: Activation function name.
    """

    embed_dim: int
    ff_dim: int
    dropout: float = 0.0
    use_bias: bool = False
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        """Apply feedforward MLP.

        Args:
            x: Input, shape (seq_len, embed_dim).
            training: Whether in training mode.

        Returns:
            Output, shape (seq_len, embed_dim).
        """
        x = nn.Dense(
            self.ff_dim,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="c_fc",
        )(x)
        if self.activation == "gelu":
            x = nn.gelu(x)
        elif self.activation == "relu":
            x = nn.relu(x)
        elif self.activation == "swish":
            x = nn.swish(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        x = nn.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="c_proj",
        )(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feedforward (GPT-2 style).

    Attributes:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        ff_dim: Feedforward hidden dimension (typically 4 * embed_dim).
        dropout: Dropout rate.
        use_bias: Whether to use bias in linear layers (default: False).
        activation: Activation function name.
    """

    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout: float = 0.0
    use_bias: bool = False
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        """Apply transformer block.

        Args:
            x: Input sequence, shape (seq_len, embed_dim).
            training: Whether in training mode.

        Returns:
            Output sequence, shape (seq_len, embed_dim).
        """
        # Self-attention with residual (pre-norm style)
        ln1_out = LayerNorm(
            self.embed_dim, use_bias=self.use_bias, name="ln_1"
        )(x)
        attn_out = CausalSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            use_bias=self.use_bias,
            name="attn",
        )(ln1_out, training=training)
        x = x + attn_out

        # Feedforward with residual (pre-norm style)
        ln2_out = LayerNorm(
            self.embed_dim, use_bias=self.use_bias, name="ln_2"
        )(x)
        mlp_out = MLP(
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            use_bias=self.use_bias,
            activation=self.activation,
            name="mlp",
        )(ln2_out, training=training)
        x = x + mlp_out

        return x


class TransformerNeuralModel(BaseNeuralModel):
    """GPT2-style transformer for dynamics modeling.

    Architecture (following NeRD/Neural Robot Dynamics):
    - Concatenate [s_t, a_t] pairs as tokens
    - Linear token embedding (vocab_size = state_dim + action_dim)
    - Add positional embeddings
    - Stack transformer blocks with causal attention
    - Separate MLP output head
    - Use final timestep representation to predict state delta

    Attributes:
        env: Environment providing model dimensions.
        embed_dim: Embedding dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        num_layers: Number of transformer blocks (default: 6).
        ff_dim: Feedforward hidden dimension (default: 4 * embed_dim).
        block_size: Maximum sequence length (default: 32).
        dropout: Dropout rate (default: 0.0 for pretraining).
        use_bias: Whether to use bias in linear layers
            (default: False, GPT-2 style).
        activation: Activation function ("gelu", "relu", or "swish").
        output_mlp_sizes: Hidden sizes for output MLP head
            (default: [64]).
    """

    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = None  # Will default to 4 * embed_dim (Optional[int])
    block_size: int = 32
    dropout: float = 0.0
    use_bias: bool = False
    activation: str = "gelu"
    output_mlp_sizes: tuple = (64,)

    @nn.compact
    def __call__(
        self, states: jax.Array, actions: jax.Array, training: bool = False
    ) -> jax.Array:
        """Predict state delta from state-action history.

        Args:
            states: State history, shape (history_length, state_dim).
            actions: Action history, shape (history_length, action_dim).
            training: Whether in training mode (affects dropout).

        Returns:
            Predicted state delta, shape (state_dim,).
        """
        seq_len = states.shape[0]
        ff_dim = self.ff_dim if self.ff_dim is not None else 4 * self.embed_dim

        # Crop sequence if longer than block_size
        if seq_len > self.block_size:
            states = states[-self.block_size:]
            actions = actions[-self.block_size:]
            seq_len = self.block_size

        # Concatenate state and action for each timestep
        # Shape: (history_length, state_dim + action_dim)
        tokens = jnp.concatenate([states, actions], axis=-1)

        # Token embeddings: Linear projection (GPT-2 style)
        # NeRD uses nn.Linear(token_dim, embed_dim) instead of embedding
        x = nn.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="wte",
        )(tokens)

        # Positional embeddings (learnable, GPT-2 style)
        pos_emb = self.param(
            "wpe",
            nn.initializers.normal(stddev=0.02),
            (self.block_size, self.embed_dim),
        )
        x = x + pos_emb[:seq_len]

        # Apply dropout (only during training)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)

        # Apply transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=ff_dim,
                dropout=self.dropout,
                use_bias=self.use_bias,
                activation=self.activation,
                name=f"h_{i}",
            )(x, training=training)

        # Final layer norm
        x = LayerNorm(self.embed_dim, use_bias=self.use_bias, name="ln_f")(x)

        # Use final timestep representation
        final_repr = x[-1]  # Shape: (embed_dim,)

        # Output MLP head (NeRD style: separate MLP after transformer)
        output_dim = self.env.model.nq + self.env.model.nv
        for i, hidden_size in enumerate(self.output_mlp_sizes):
            final_repr = nn.Dense(
                hidden_size,
                use_bias=self.use_bias,
                kernel_init=nn.initializers.normal(stddev=0.02),
                name=f"output_mlp_{i}",
            )(final_repr)
            if self.activation == "gelu":
                final_repr = nn.gelu(final_repr)
            elif self.activation == "relu":
                final_repr = nn.relu(final_repr)
            elif self.activation == "swish":
                final_repr = nn.swish(final_repr)

        # Final output projection
        delta = nn.Dense(
            output_dim,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="output",
        )(final_repr)

        return delta

