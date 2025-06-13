# **Bonsai: The Underlying Model**

This document outlines the technical details of the probabilistic model and algorithm used in the `Bonsai` software package. Bonsai constructs a tree-based representation of high-dimensional data in a way that is mathematically robust, parameter-free, and distortion-free.

## **1. Overview**

The core objective of Bonsai is to find the most probable tree `T` that explains a set of high-dimensional data points `{x_i}`. It achieves this using a Bayesian framework, defining a posterior probability for a tree and then searching for the tree that maximizes this probability.

The model is generative, meaning it defines a hypothetical process by which a tree could have generated the observed data.

## **2. Model Inputs**

Bonsai does not operate on raw count matrices directly. It requires two inputs, which are derived from a probabilistic dimensionality reduction pre-processing step (e.g., Probabilistic Principal Component Analysis - PPCA).

1.  **`x_i` (Measured Positions):** A vector representing the position of each cell `i` in a `D`-dimensional latent space (e.g., `D=50`).
2.  **`σ_i` (Feature Error Bars):** A vector representing the standard deviation (uncertainty) of each cell's position `x_i` along each of the `D` dimensions. This is a critical input that quantifies the measurement noise for each cell individually.

The conceptual flow is as follows:

```
[Raw Gene Expression Matrix] -> [Probabilistic PCA] -> {x_i, σ_i} -> [Bonsai Algorithm] -> [Optimal Tree T]
```

## **3. The Probabilistic Model**

The model finds the best tree by maximizing the posterior probability `P(T | {x_i})`. Using Bayes' theorem, this is expressed as:

`P(T | {x_i}) ∝ P({x_i} | T) * P(T)`

Where:
*   `P(T | {x_i})` is the **Posterior**: The probability of the tree given the data (our objective).
*   `P({x_i} | T)` is the **Likelihood**: The probability of observing the data, given a tree.
*   `P(T)` is the **Prior**: The intrinsic probability of a tree, independent of the data.

### **3.1. The Likelihood: `P({x_i} | T)`**

The likelihood function connects the abstract tree `T` to the measured data `{x_i}`. It's based on the assumption that each measured data point `x_i` is a noisy observation of a "true" point `y_i` that lies perfectly on the tree `T`.

The model assumes this noise is Gaussian:

`x_i ~ N(y_i, σ_i^2)`

Where `N` is a `D`-dimensional Gaussian distribution centered at the true position `y_i` with variance `σ_i^2`.

The total likelihood `P({x_i} | T)` is calculated by marginalizing (integrating over) all possible latent positions `{y_i}` on the tree. This integral can be solved analytically. In essence, this likelihood function evaluates how well the cloud of data points `{x_i}`, considering their individual error bars `{σ_i}`, can be explained by being "tethered" to the scaffold of tree `T`.

### **3.2. The Prior: `P(T)`**

The prior assigns a probability to a tree structure itself, acting as a form of regularization. It enforces a preference for simpler, more "parsimonious" trees.

The prior is defined over the total branch length of the tree. It assumes that the length `l_b` of any given branch `b` is drawn from an exponential distribution:

`P(l_b) ∝ exp(-λ * l_b)`

The prior for the entire tree `P(T)` is the product of the priors for all its branches. This penalizes overly complex trees with many long branches, preventing the model from overfitting the noise in the data. The hyperparameter `λ` is set to a default value based on standard phylogenetic practice and is not user-tunable.

## **4. The Algorithm: Finding the Maximum a Posteriori (MAP) Tree**

The space of all possible tree topologies for `N` cells is immense (`(2N - 3)!!`), making an exhaustive search impossible. Bonsai uses a stochastic search algorithm to find the tree that maximizes the posterior probability (`P(T | {x_i})`).

1.  **Initialization**: The algorithm begins with a randomly generated or simple starting tree (e.g., a star-like tree).

2.  **Iterative Search**: The algorithm iteratively proposes changes to the current tree topology and branch lengths. The primary move used for exploring different topologies is the **Subtree Prune and Regraft (SPR)** operation:
    *   A random branch in the tree is cut, splitting the tree into a subtree and a main tree.
    *   The subtree is then re-attached (regrafted) to a different edge on the main tree.

3.  **Evaluation**: For each newly proposed tree `T_new`, the posterior probability `P(T_new | {x_i})` is calculated.

4.  **Acceptance/Rejection**: The new tree is accepted or rejected based on a Metropolis-Hastings criterion. If `P(T_new) > P(T_current)`, the move is always accepted. If it is worse, it may still be accepted with a small probability, allowing the search to escape local optima.

5.  **Convergence**: This process is repeated for many iterations until the algorithm converges on a stable, high-probability tree. The final result is the **Maximum a Posteriori (MAP)** tree—the single tree with the highest posterior probability found during the search.

This final tree `T` is the Bonsai representation. The distances between any two cells, when measured along the paths of this tree, provide a distortion-free estimate of their true separation in the high-dimensional space.