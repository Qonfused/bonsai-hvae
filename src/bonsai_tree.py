import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from collections import defaultdict

from .hyperbolic_geometry import geodesic_dist, mobius_add
from .bonsai_felsenstein import felsenstein_pruning


class Tree:
  def __init__(self, parents, children, node_positions, leaf_ids, root):
    self.parents = parents
    self.children = children
    self.node_positions = node_positions
    self.leaf_ids = leaf_ids
    self.node_ids = list(node_positions.keys())
    self.root = root

  def get_path(self, a, b):
    path_a = []
    ta = a
    while ta in self.parents:
      path_a.append(ta)
      ta = self.parents[ta]
    path_a.append(ta)
    tb = b
    path_b = []
    while tb in self.parents:
      path_b.append(tb)
      tb = self.parents[tb]
    path_b.append(tb)
    i = len(path_a) - 1
    j = len(path_b) - 1
    while i >= 0 and j >= 0 and path_a[i] == path_b[j]:
      i -= 1
      j -= 1
    path = path_a[:i+2] + path_b[j+1::-1]
    return path

  def tree_geodesic(self, a, b):
    path = self.get_path(a, b)
    dist = 0.0
    for u, v in zip(path[:-1], path[1:]):
      dist += float(geodesic_dist(self.node_positions[u], self.node_positions[v]))
    return dist

  def pairwise_tree_dist_matrix(self):
    N = len(self.leaf_ids)
    D = np.zeros((N, N))
    for i in range(N):
      for j in range(N):
        D[i, j] = self.tree_geodesic(self.leaf_ids[i], self.leaf_ids[j])
    return D

def initialize_random_binary_tree(mu):
  """
  Randomly constructs a binary tree with N leaves, each leaf node initialized at mu[i].
  Returns Tree object.
  """
  N, latent_dim = mu.shape
  node_positions = {}
  parents = {}
  children = defaultdict(list)
  leaf_ids = []
  curr_id = 0
  # Create a list of current nodes (leaves first)
  curr_nodes = []
  for i in range(N):
    node_positions[curr_id] = mu[i]
    leaf_ids.append(curr_id)
    curr_nodes.append(curr_id)
    curr_id += 1

  # Randomly join nodes until one remains (root)
  rng = np.random.default_rng()
  while len(curr_nodes) > 1:
    a, b = rng.choice(curr_nodes, 2, replace=False)
    new_pos = (node_positions[a] + node_positions[b]) / 2
    node_positions[curr_id] = new_pos
    children[curr_id] = [a, b]
    parents[a] = curr_id
    parents[b] = curr_id
    curr_nodes = [n for n in curr_nodes if n not in [a, b]] + [curr_id]
    curr_id += 1

  root = curr_nodes[0]
  return Tree(parents, children, node_positions, leaf_ids, root=root)

def tree_log_prior(tree, lam=0.1):
  length = 0.0
  for node in tree.node_ids:
    if node in tree.parents:
      parent = tree.parents[node]
      length += float(geodesic_dist(tree.node_positions[node], tree.node_positions[parent]))
  return -lam * length

def random_spr(tree, key):
  """
  Subtree Prune and Regraft (SPR) for binary tree.
  1. Pick random non-root node to prune.
  2. Remove the subtree rooted at that node.
  3. Pick random edge elsewhere to regraft.
  """
  non_root_nodes = [n for n in tree.node_ids if n != tree.root]
  if len(non_root_nodes) < 2:
    return tree  # Can't SPR a tree with only two leaves
  rng = np.random.default_rng()
  prune_node = rng.choice(non_root_nodes)
  prune_parent = tree.parents[prune_node]
  # Remove prune_node from parent's children
  new_children = defaultdict(list, {k: list(v) for k, v in tree.children.items()})
  new_children[prune_parent] = [c for c in new_children[prune_parent] if c != prune_node]
  # Remove parent link
  new_parents = dict(tree.parents)
  del new_parents[prune_node]

  # Find all possible regraft locations (avoid descendants of prune_node)
  def descendants(node):
    result = set()
    stack = [node]
    while stack:
      n = stack.pop()
      result.add(n)
      stack.extend(tree.children.get(n, []))
    return result
  descendants_of_prune = descendants(prune_node)
  possible_edges = [(n, c) for n in tree.children for c in tree.children[n]
            if n not in descendants_of_prune and c not in descendants_of_prune]
  if not possible_edges:
    return tree  # No valid regraft
  regraft_parent, regraft_child = rng.choice(possible_edges)
  # Regraft: insert prune_node between regraft_parent and regraft_child
  new_children[regraft_parent] = [prune_node if c == regraft_child else c for c in new_children[regraft_parent]]
  new_parents[prune_node] = regraft_parent
  new_children[prune_node] = [regraft_child]
  new_parents[regraft_child] = prune_node
  # Copy node positions, leaf_ids, root
  return Tree(new_parents, new_children, tree.node_positions, tree.leaf_ids, tree.root)

def random_geometric_move(tree, key, step_size=0.05):
  """
  Perturb a random internal (non-leaf) node in the Poincaré disk.
  """
  internal_nodes = [n for n in tree.node_ids if n not in tree.leaf_ids]
  if not internal_nodes:
    return tree
  rng = np.random.default_rng()
  node = rng.choice(internal_nodes)
  latent_dim = tree.node_positions[node].shape[0]
  noise = random.normal(key, (latent_dim,)) * step_size
  from hyperbolic_geometry import mobius_add
  new_pos = mobius_add(tree.node_positions[node], noise)
  new_node_positions = dict(tree.node_positions)
  new_node_positions[node] = new_pos
  return Tree(tree.parents, tree.children, new_node_positions, tree.leaf_ids, tree.root)

def bonsai_mcmc_search(mu, sigma, n_iter=100, lam=0.1, latent_dim=2, key=None):
  """
  Full MCMC search for MAP tree over binary trees.
  """
  N = mu.shape[0]
  if key is None:
    key = random.PRNGKey(0)
  tree = initialize_random_binary_tree(mu)
  best_tree = tree
  best_score = -np.inf

  for iter in range(n_iter):
    key, subkey = random.split(key)
    move_type = random.bernoulli(subkey, 0.5)
    if move_type:
      proposal = random_geometric_move(tree, key)
    else:
      proposal = random_spr(tree, key)
    # Compute log posterior
    ll = felsenstein_pruning(proposal, mu, sigma)
    lp = tree_log_prior(proposal, lam)
    score = ll + lp
    old_ll = felsenstein_pruning(tree, mu, sigma)
    old_lp = tree_log_prior(tree, lam)
    old_score = old_ll + old_lp
    accept = np.log(np.random.uniform()) < (score - old_score)
    if accept:
      tree = proposal
      if score > best_score:
        best_tree = tree
        best_score = score
  D = best_tree.pairwise_tree_dist_matrix()
  return best_tree, D
