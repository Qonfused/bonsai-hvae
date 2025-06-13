import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import haiku as hk
import optax

from .hvae import HVAE
from .losses import reconstruction_loss, kl_wrapped_normal, bonsai_structural_loss
from .bonsai_tree import bonsai_mcmc_search
from .training_utils import linear_anneal


class BonsaiHVAETrainer:
  def __init__(self, latent_dim, data_dim, binary=True, kl_anneal_epochs=50, bonsai_anneal_epochs=50,
         bonsai_update_freq=5, alpha_max=1.0, seed=42):
    self.latent_dim = latent_dim
    self.data_dim = data_dim
    self.binary = binary
    self.kl_anneal_epochs = kl_anneal_epochs
    self.bonsai_anneal_epochs = bonsai_anneal_epochs
    self.bonsai_update_freq = bonsai_update_freq
    self.alpha_max = alpha_max
    self.key = random.PRNGKey(seed)

    def forward(x, key):
      hvae = HVAE(latent_dim, data_dim)
      return hvae(x, key)
    self.forward = hk.transform(forward)
    self.opt = optax.adam(1e-3)

  def train(self, X, num_epochs=100, batch_size=32, verbose=True):
    N = X.shape[0]
    params = self.forward.init(self.key, X[:batch_size], self.key)
    opt_state = self.opt.init(params)
    step = 0

    tree, D_target = None, jnp.zeros((N, N))

    for epoch in range(num_epochs):
      beta = linear_anneal(0.0, 1.0, epoch, self.kl_anneal_epochs)
      alpha = linear_anneal(0.0, self.alpha_max, epoch, self.bonsai_anneal_epochs)

      if epoch % self.bonsai_update_freq == 0 or tree is None:
        mu_list, sigma_list = [], []
        for i in range(0, N, batch_size):
          batch = X[i:i+batch_size]
          key = random.fold_in(self.key, step)
          mu, sigma = hk.without_apply_rng(hk.transform(lambda x: HVAE(self.latent_dim, self.data_dim).encode(x))).apply(params, batch)
          mu_list.append(mu)
          sigma_list.append(sigma)
        mu_all = jnp.concatenate(mu_list, axis=0)
        sigma_all = jnp.concatenate(sigma_list, axis=0)
        tree, D_target = bonsai_mcmc_search(mu_all, sigma_all, latent_dim=self.latent_dim, key=random.fold_in(self.key, epoch))

      idx = np.random.permutation(N)
      X_shuff = X[idx]
      for i in range(0, N, batch_size):
        batch = X_shuff[i:i+batch_size]
        key = random.fold_in(self.key, step)
        def loss_fn(params):
          x_rec, mu, sigma, z = self.forward.apply(params, key, batch, key)
          rec = jnp.mean(reconstruction_loss(batch, x_rec, binary=self.binary))
          kl = jnp.mean(kl_wrapped_normal(mu, sigma))
          bonsai_loss = bonsai_structural_loss(mu, D_target[idx[i:i+batch_size]][:, idx[i:i+batch_size]])
          return rec + beta * kl + alpha * bonsai_loss
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        step += 1

      if verbose:
        print(f"Epoch {epoch+1}/{num_epochs}: beta={beta:.3f}, alpha={alpha:.3f}")

    return params, tree
