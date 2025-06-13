import haiku as hk
import jax
import jax.numpy as jnp


def encoder_mlp(x, latent_dim):
  mlp = hk.Sequential([
    hk.Linear(128), jax.nn.relu,
    hk.Linear(128), jax.nn.relu,
    hk.Linear(2 * latent_dim),
  ])
  out = mlp(x)
  mu_pre = out[..., :latent_dim]
  logvar = out[..., latent_dim:]
  return mu_pre, logvar

def decoder_mlp(z_euclid, data_dim):
  mlp = hk.Sequential([
    hk.Linear(128), jax.nn.relu,
    hk.Linear(128), jax.nn.relu,
    hk.Linear(data_dim),
  ])
  return mlp(z_euclid)
