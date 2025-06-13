import jax
import jax.numpy as jnp

from .hyperbolic_geometry import geodesic_dist


def reconstruction_loss(x, x_rec, binary=True):
  if binary:
    return -jnp.sum(x * jax.nn.log_sigmoid(x_rec) + (1 - x) * jax.nn.log_sigmoid(-x_rec), axis=-1)
  else:
    return jnp.sum((x - x_rec) ** 2, axis=-1)

def kl_wrapped_normal(mu, sigma, eps=1e-5):
  mu2 = jnp.sum(mu ** 2, axis=-1)
  sigma2 = jnp.sum(sigma ** 2, axis=-1)
  return 0.5 * (mu2 + sigma2 - jnp.log(sigma2 + eps) - mu.shape[-1])

def bonsai_structural_loss(mu, D_target):
  N = mu.shape[0]
  dists = jax.vmap(lambda x: jax.vmap(lambda y: geodesic_dist(x, y))(mu))(mu)
  loss = jnp.mean((dists - D_target) ** 2)
  return loss
