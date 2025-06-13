import jax
import jax.numpy as jnp


def linear_anneal(start, end, step, total_steps):
  alpha = (step / total_steps)
  return start + (end - start) * jnp.clip(alpha, 0.0, 1.0)
