import jax
import jax.numpy as jnp
import jax.random as random

from .hyperbolic_geometry import exp_map


def sample_wrapped_normal(key, mu, sigma):
  """Sample from a wrapped normal centered at mu, with std sigma in tangent space."""
  eps = random.normal(key, sigma.shape)
  v = eps * sigma
  return exp_map(mu, v)