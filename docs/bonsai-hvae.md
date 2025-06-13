# **Bonsai-HVAE: Hierarchical Representation Learning with Hyperbolic VAEs and Bayesian Tree Inference**

This document outlines the technical details of the generative model and algorithm implemented in the `bonsai-hvae` software package. Bonsai-HVAE is designed to learn, represent, and infer hierarchical structure in high-dimensional data by combining the geometric flexibility of a hyperbolic VAE (H-VAE) with the statistical rigor of Bayesian tree inference (Bonsai). The model alternates between learning a geometry-aware embedding and optimizing a tree structure, using a carefully managed co-training scheme for stability and interpretability.

## **1. Overview**

The core objective of Bonsai-HVAE is to discover the most probable tree `T` that explains a collection of high-dimensional data points `{x_i}`. This is achieved by embedding each data point into a hyperbolic latent space using a VAE, and then searching for the tree that best explains these latent representations. The model employs a Bayesian framework, maximizing the posterior probability of a tree given the latent encodings. 

By leveraging the natural geometric alignment between hyperbolic spaces and tree-like data, Bonsai-HVAE overcomes the limitations of traditional VAEs, which are inherently Euclidean and poorly suited for hierarchical data.

## **2. Model Inputs**

Bonsai-HVAE does not operate directly on raw data such as count matrices. Instead, it processes the data through the H-VAE encoder, which produces two crucial outputs for each input:

1.  **`μ_i` (Latent Mean):** The mean vector for each data point, representing its position in the Poincaré disk (hyperbolic latent space).
2.  **`σ_i` (Latent Uncertainty):** A vector of standard deviations, representing uncertainty for each coordinate of `μ_i` in the tangent space at `μ_i`.

The conceptual workflow is:

```
[Raw Data] -> [H-VAE Encoder] -> {μ_i, σ_i} -> [Bonsai Algorithm] -> [Optimal Tree T]
```

This separation allows Bonsai-HVAE to learn a latent geometry that is amenable to tree-based inference, while also quantifying the uncertainty in each embedding.

## **3. The Probabilistic Model**

At the heart of Bonsai-HVAE is a generative model that seeks the tree `T` that maximizes the posterior probability given the latent means and uncertainties:

```
P(T | Z) ∝ P(Z | T) * P(T)
```
where `Z = {μ_i, σ_i}` are the latent codes and uncertainties for all samples.

- **Posterior (`P(T | Z)`):** The probability of a tree given the latent encodings.
- **Likelihood (`P(Z | T)`):** The probability of observing the latent encodings, given a tree.
- **Prior (`P(T)`):** The intrinsic probability of a tree, independent of the data.

### **3.1. Hyperbolic Variational Autoencoder (H-VAE)**

The H-VAE is responsible for mapping high-dimensional data to a hyperbolic latent space:

#### **Encoder (`q_φ(z|x)`):**

- Each input `x` is encoded using a neural network (MLP) into a pre-image mean `μ'_x` and log-variance `log(σ^2_x)`.
- The mean is projected into the Poincaré disk: `μ_x = tanh(μ'_x)`, ensuring `||μ_x|| < 1`.
- The variance is exponentiated: `σ_x = exp(0.5 * log(σ^2_x))`.
- The resulting parameters `(μ_x, σ_x)` define a wrapped normal distribution in hyperbolic space.

#### **Reparameterization (Sampling in Hyperbolic Space):**

- To sample a latent code `z`, we:
  1. Draw `ε ~ N(0, I)` in the tangent space at `μ_x`.
  2. Compute `v = ε * σ_x`.
  3. Project `v` onto the manifold via the exponential map:
     ```
     z = exp_μx(v) = μ_x ⊕ (tanh((λ_μx * ||v||)/2) * v / ||v||)
     ```
     where `⊕` is Möbius addition and `λ_μx = 2/(1 - ||μ_x||^2)`.

#### **Decoder (`p_θ(x|z)`):**

- The decoder projects `z` back to the tangent space at the origin using the logarithmic map:
  ```
  v_euclidean = log_0(z) = tanh^(-1)(||z||) * z / ||z||
  ```
- This Euclidean vector is decoded by a neural network to reconstruct the input.

#### **Objective Function:**

The H-VAE is trained by maximizing the evidence lower bound (ELBO):

```
L_HVAE = E_qφ(z|x)[ -log pθ(x|z) ] + β * D_KL( qφ(z|x) || p(z) )
```
- The reconstruction term measures fidelity to the input.
- The KL-divergence encourages regularity in the latent space.
- The annealing parameter `β` is gradually increased during training to prevent posterior collapse.

### **3.2. The Bonsai Tree Model**

The Bonsai component finds the tree that best explains the latent codes output by the H-VAE.

#### **Likelihood (`P(Z | T)`):**

- Each observed latent mean `μ_i` is modeled as a noisy observation of a "true" latent position `y_i` on the tree:
  ```
  μ_i ~ WrappedNormal(y_i, σ_i)
  ```
- The likelihood is computed by integrating over all possible true positions `{y_i}` on the tree, using a dynamic programming algorithm (adapted Felsenstein's pruning algorithm) in hyperbolic space.

#### **Prior (`P(T)`):**

- The prior penalizes the total branch length of the tree, measured as the sum of geodesic distances in the Poincaré disk:
  ```
  P(T) ∝ exp(-λ * L(T)),    L(T) = Σ_{(u,v) ∈ Edges(T)} d_hyperbolic(p_u, p_v)
  ```

#### **MCMC Search for MAP Tree:**

- The MAP tree is found using a Metropolis-Hastings MCMC algorithm:
  1. **Initialization:** Begin with a simple or random tree.
  2. **Propose Moves:** Alternate between subtree prune-and-regraft (SPR) moves (changing topology) and geometric moves (perturbing internal node positions).
  3. **Evaluate:** Compute the posterior probability for each candidate tree.
  4. **Accept/Reject:** Accept new proposals probabilistically according to the posterior ratio.
  5. **Converge:** Repeat until a high-probability tree is found.

## **4. Co-Training and Optimization**

A key innovation in Bonsai-HVAE is the **annealed alternating optimization** of the H-VAE and Bonsai components, ensuring both stability and interpretability.

### **Co-Training Workflow:**

1. **H-VAE Pre-training:** The encoder and decoder are trained alone for several epochs, learning an initial latent embedding without structural constraints.
2. **Alternating Optimization:**
   - **Bonsai Update (Infrequent):** Use current latent means and uncertainties to infer a new tree and compute a target distance matrix (`D_target`).
   - **H-VAE Update (Frequent):** Train the H-VAE for several epochs, optimizing a joint loss:
     ```
     L_Total = L_HVAE + α * L_Bonsai
     ```
     where
     ```
     L_Bonsai = MSE( d_hyperbolic(z_i, z_j), D_target[i, j] )
     ```
     aligns pairwise geodesic distances in latent space with those on the tree.
3. **Annealing Schedules:**
   - **KL Annealing (β):** Gradually increase the weight of the KL term from 0 to 1 to avoid posterior collapse.
   - **Bonsai Coupling Annealing (α):** Gradually increase the structural loss weight from 0 to `α_max`, preventing abrupt imposition of tree constraints.
   - **Update Frequency:** The H-VAE is updated continuously, while Bonsai inference is performed every few epochs (e.g., every 5–10 epochs).

The co-training cycle continues until validation metrics stabilize, yielding both a quality embedding and a meaningful tree.

## **5. Summary**

Bonsai-HVAE provides a principled framework for hierarchical representation learning, combining the geometric advantages of hyperbolic latent spaces with the statistical rigor of Bayesian tree inference. By alternating and annealing the optimization of both model components, Bonsai-HVAE produces embeddings that are faithful to the data, robust to uncertainty, and structurally interpretable.

---

## **Appendix: Implementation Notes and Technical Details**

This appendix provides additional technical and practical details required for a faithful implementation of Bonsai-HVAE.

### **A. H-VAE Encoder Outputs and Projections**

- The encoder produces a pre-image mean `μ'_x` and log-variance `log(σ²_x)` for each input.
- The latent mean `μ_x` is projected to the Poincaré disk via `μ_x = tanh(μ'_x)`, ensuring `||μ_x|| < 1`.
  - In practice, it is recommended to clip `||μ_x||` so that `||μ_x|| < 1 - ε`, where `ε` is a small constant (e.g., `1e-5`), to avoid numerical instability near the boundary of the disk.
- The standard deviation vector is computed as `σ_x = exp(0.5 * log(σ²_x))`, ensuring positivity and stability.

### **B. Wrapped Normal Sampling and Exponential Map**

- The reparameterization trick for hyperbolic space proceeds as:
  1. Sample Euclidean noise `ε ~ N(0, I)` in the tangent space at `μ_x`.
  2. Compute `v = ε * σ_x`.
  3. Project `v` to the manifold using the exponential map:
     ```
     z = exp_μx(v) = μ_x ⊕ (tanh((λ_μx * ||v||)/2) * v / ||v||)
     ```
     where `⊕` denotes Möbius addition, and `λ_μx = 2/(1 - ||μ_x||²)`.
- All operations in this pipeline (sampling, mapping) must be differentiable for backpropagation.

### **C. Tree Likelihood and Felsenstein’s Pruning in Hyperbolic Space**

- The likelihood for each tree is calculated by integrating over possible true latent positions `{y_i}` on the tree, using a dynamic programming approach (Felsenstein’s pruning algorithm).
- This algorithm is adapted for hyperbolic space:
  - All distances are geodesic distances in the Poincaré disk.
  - Likelihoods are computed using wrapped normal distributions for each `(μ_i, σ_i)`.
  - Internal node positions are optimized as continuous points in hyperbolic space, not just at discrete locations.

### **D. Annealing Schedules**

- Both `β` (KL-divergence strength) and `α` (Bonsai coupling) are annealed from 0 to their final values over a user-specified schedule, which may be linear or sigmoidal.
- The schedule is typically defined over a fixed number of epochs or training steps.
- Early stopping or patience-based criteria can be used to determine convergence.

### **E. Update Frequency and Decoupling**

- The H-VAE is trained continuously on every batch, while the Bonsai search for an updated tree is performed only every few epochs (e.g., every 5–10 epochs).
- This decoupling provides the H-VAE with a fixed structural target (`D_target`) for several epochs, improving optimization stability and computational feasibility.
- The Bonsai loss operates on a fixed pairwise geodesic distance matrix derived from the current MAP tree.

### **F. Numerical Stability**

- All latent means are clipped to remain strictly inside the Poincaré disk (`||μ_x|| < 1 - ε`).
- Variances are initialized to small values and learning rates are tuned to avoid instability, especially near the boundary of the disk.

### **G. Loss Function Details**

- The Bonsai structural loss (`L_Bonsai`) is defined as the mean squared error (MSE) between all pairwise geodesic distances in the H-VAE latent space and the corresponding distances in the MAP tree.
- The reconstruction loss within the H-VAE is chosen to match the data type:
  - Use binary cross-entropy for binary data.
  - Use mean squared error for continuous data.

### **H. Optimization Details**

- The Adam optimizer is recommended for both encoder and decoder parameters.
- Gradients from the Bonsai loss are backpropagated through the encoder only, since the tree structure is fixed during each H-VAE update cycle and is not differentiable.

### **I. Hyperparameter Ranges**

- Typical choices are:
  - Latent dimension (`n`): 2–10.
  - KL and Bonsai annealing schedules: 10–100 epochs.
  - Bonsai tree update frequency: every 5–10 epochs.
- Learning rates, annealing schedules, and other optimizer parameters are set empirically based on validation performance.

---

This appendix is intended as a reference for both the mathematical formulation and practical implementation of Bonsai-HVAE, ensuring reproducibility and robustness.