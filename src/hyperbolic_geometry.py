import jax
import jax.numpy as jnp


def mobius_add(x, y, eps=1e-5):
  x2 = jnp.sum(x * x, axis=-1, keepdims=True)
  y2 = jnp.sum(y * y, axis=-1, keepdims=True)
  xy = jnp.sum(x * y, axis=-1, keepdims=True)
  num = (1 + 2 * xy + y2) * x + (1 - x2) * y
  denom = 1 + 2 * xy + x2 * y2
  return num / (denom + eps)

def lambda_x(x, eps=1e-5):
  return 2. / (1 - jnp.sum(x * x, axis=-1, keepdims=True) + eps)

def exp_map(x, v, eps=1e-5):
  norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True) + eps
  lam = lambda_x(x, eps)
  second_term = jnp.tanh(lam * norm_v / 2) * (v / norm_v)
  return mobius_add(x, second_term)

def log_map_zero(y, eps=1e-5):
  norm_y = jnp.linalg.norm(y, axis=-1, keepdims=True) + eps
  return jnp.arctanh(norm_y) * (y / norm_y)

def geodesic_dist(x, y, eps=1e-5):
  diff = mobius_add(-x, y)
  norm = jnp.clip(jnp.linalg.norm(diff, axis=-1), 0, 1 - eps)
  return 2 * jnp.arctanh(norm)

def project_to_disk(x, eps=1e-5):
  norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
  return x * (1 - eps) / jnp.maximum(norm, 1 - eps)
