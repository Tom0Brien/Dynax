"""GPT2-style transformer architecture for dynamics modeling."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from dynax.base import BaseDynamicsModel


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer (GPT2 style).

    Attributes:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate (0.0 to disable).
    """

    embed_dim: int
    num_heads: int
    dropout: float = 0.0

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

        # Compute Q, K, V
        qkv = nn.Dense(self.embed_dim * 3, name="qkv")(x)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jnp.einsum("thd,shd->ts", q, k) * scale

        # Causal mask: only attend to past and current timesteps
        mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
        attn_weights = jnp.where(mask, -jnp.inf, attn_weights)
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Apply dropout
        if self.dropout > 0.0 and training:
            attn_weights = nn.Dropout(self.dropout)(
                attn_weights, deterministic=not training
            )

        # Apply attention to values
        out = jnp.einsum("ts,shd->thd", attn_weights, v)
        out = out.reshape(seq_len, self.embed_dim)

        # Output projection
        out = nn.Dense(self.embed_dim, name="out_proj")(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feedforward.

    Attributes:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        ff_dim: Feedforward hidden dimension.
        dropout: Dropout rate.
        activation: Activation function name.
    """

    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout: float = 0.0
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
        # Self-attention with residual
        residual = x
        x = nn.LayerNorm(name="ln1")(x)
        x = CausalSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            name="attn",
        )(x, training=training)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = x + residual

        # Feedforward with residual
        residual = x
        x = nn.LayerNorm(name="ln2")(x)
        x = nn.Dense(self.ff_dim, name="ff1")(x)
        if self.activation == "gelu":
            x = nn.gelu(x)
        elif self.activation == "relu":
            x = nn.relu(x)
        elif self.activation == "swish":
            x = nn.swish(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.embed_dim, name="ff2")(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = x + residual

        return x


class TransformerDynamicsModel(BaseDynamicsModel):
    """GPT2-style transformer for dynamics modeling.

    Architecture:
    - Embed each [s_t, a_t] pair as a token
    - Add positional embeddings
    - Stack transformer blocks with causal attention
    - Use final timestep representation to predict state delta

    Attributes:
        env: Environment providing model dimensions.
        embed_dim: Embedding dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        num_layers: Number of transformer blocks (default: 6).
        ff_dim: Feedforward hidden dimension (default: 1024).
        dropout: Dropout rate (default: 0.1).
        activation: Activation function ("gelu", "relu", or "swish").
    """

    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"

    @nn.compact
    def __call__(self, states: jax.Array, actions: jax.Array) -> jax.Array:
        """Predict state delta from state-action history.

        Args:
            states: State history, shape (history_length, state_dim).
            actions: Action history, shape (history_length, action_dim).

        Returns:
            Predicted state delta, shape (state_dim,).
        """
        seq_len = states.shape[0]

        # Concatenate state and action for each timestep
        # Shape: (history_length, state_dim + action_dim)
        tokens = jnp.concatenate([states, actions], axis=-1)

        # Token embeddings: project to embed_dim
        x = nn.Dense(self.embed_dim, name="token_embed")(tokens)

        # Positional embeddings (learnable)
        pos_emb = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (self.history_length, self.embed_dim),
        )
        x = x + pos_emb[:seq_len]

        # Apply transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                activation=self.activation,
                name=f"transformer_block_{i}",
            )(x, training=False)  # No dropout during inference

        # Layer norm before output
        x = nn.LayerNorm(name="final_ln")(x)

        # Use final timestep representation to predict delta
        final_repr = x[-1]  # Shape: (embed_dim,)

        # Output projection to state delta
        output_dim = self.env.model.nq + self.env.model.nv
        delta = nn.Dense(output_dim, name="output")(final_repr)

        return delta

