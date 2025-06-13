import haiku as hk
import jax
import jax.numpy as jnp

from .hyperbolic_geometry import project_to_disk, log_map_zero
from .mlp_modules import encoder_mlp, decoder_mlp
from .wrapped_normal import sample_wrapped_normal


class HVAE(hk.Module):
  def __init__(self, latent_dim, data_dim, eps=1e-5):
    super().__init__()
    self.latent_dim = latent_dim
    self.data_dim = data_dim
    self.eps = eps

  def encode(self, x):
    mu_pre, logvar = encoder_mlp(x, self.latent_dim)
    mu = jnp.tanh(mu_pre)
    mu = project_to_disk(mu, self.eps)
    sigma = jnp.exp(0.5 * logvar)
    return mu, sigma

  def sample_latent(self, key, mu, sigma):
    return sample_wrapped_normal(key, mu, sigma)

  def decode(self, z):
    z_euclid = log_map_zero(z)
    return decoder_mlp(z_euclid, self.data_dim)

  def __call__(self, x, key):
    mu, sigma = self.encode(x)
    z = self.sample_latent(key, mu, sigma)
    x_rec = self.decode(z)
    return x_rec, mu, sigma, z
