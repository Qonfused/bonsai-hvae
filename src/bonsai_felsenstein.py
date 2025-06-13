import jax.numpy as jnp
import numpy as np
from collections import defaultdict

from .hyperbolic_geometry import geodesic_dist


def postorder_traversal(children, root):
  """Return postorder list of node ids."""
  order = []
  def recurse(node):
    for child in children[node]:
      recurse(child)
    order.append(node)
  recurse(root)
  return order

def felsenstein_pruning(tree, leaf_obs_mu, leaf_obs_sigma):
  """
  Compute log-likelihood of observed data at leaves under a binary tree with hyperbolic branch lengths.
  tree: Tree object
  leaf_obs_mu: [N, d] latent means (leaves)
  leaf_obs_sigma: [N, d] latent stds (leaves)
  Returns: log-marginal likelihood (scalar)
  """
  # Assumptions:
  # - Each node emits a wrapped normal (Euclidean approx)
  # - At each internal node, integrate over possible positions; for now, use a "local" approximation by maximizing
  #   (for full marginal, use numerical quadrature or message passing—see literature for details)
  # This is a simplified, tractable version for demonstration.

  # Map leaf indices to node ids
  leaf_ids = tree.leaf_ids
  node_positions = tree.node_positions
  children = tree.children
  parents = tree.parents

  N, d = leaf_obs_mu.shape
  log_likelihoods = {}

  # Compute at leaves: likelihood of observed mu given true (node) mu and sigma
  for i, node in enumerate(leaf_ids):
    obs_mu = leaf_obs_mu[i]
    obs_sigma = leaf_obs_sigma[i]
    node_mu = node_positions[node]
    # Euclidean approx: Normal PDF
    diff = obs_mu - node_mu
    quad = np.sum((diff / (obs_sigma + 1e-8)) ** 2)
    log_det = d * np.log(obs_sigma + 1e-8)
    log_likelihoods[node] = -0.5 * (d * np.log(2 * np.pi) + log_det + quad)

  # Upward pass: at each internal node, combine log-likelihoods from children
  post_order = postorder_traversal(children, tree.root)
  for node in post_order:
    if node in leaf_ids:
      continue
    child_nodes = children[node]
    # For demonstration, assume node position is fixed (from node_positions)
    node_mu = node_positions[node]
    # Sum log-likelihoods from children, adding branch penalty (geodesic)
    ll = 0.0
    for c in child_nodes:
      branch_len = float(geodesic_dist(node_mu, node_positions[c]))
      ll += log_likelihoods[c] - branch_len  # branch penalty in log domain
    log_likelihoods[node] = ll

  # Root node
  root = tree.root
  return float(log_likelihoods[root])
