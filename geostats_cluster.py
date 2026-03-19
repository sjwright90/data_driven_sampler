"""
Geostatistical Similarity Matrix
=================================
Implementation of the non-parametric kernel-based geostatistical similarity
measure from Fouedjio (2016a,b,c, 2017a,b).

Given:
    - coords: (n, d) array of spatial coordinates (2D or 3D)
    - Z: (n, p) array of standardised feature variables

Produces:
    - S: (n, n) similarity matrix where S[t, t'] measures spatially-informed
      similarity between sample locations x_t and x_t'

The similarity is derived from non-parametric kernel estimates of the direct
and cross-variograms, using an Epanechnikov kernel with compact support.

References:
    Fouedjio, F. (2016). A hierarchical clustering method for multivariate
    geostatistical data. Spatial Statistics, 18, 333-351.

    Fouedjio, F. (2017). A clustering approach for discovering intrinsic
    clusters in multivariate geostatistical data. In Machine Learning and
    Knowledge Discovery in Databases (pp. 491-500). Springer.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, Tuple
import warnings


def epanechnikov_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Epanechnikov kernel: K(h) ∝ (λ² - ||h||²) for ||h|| <= λ, else 0.

    Parameters
    ----------
    distances : np.ndarray
        Array of Euclidean distances.
    bandwidth : float
        Bandwidth parameter λ (support radius).

    Returns
    -------
    np.ndarray
        Kernel weights (unnormalised). Zero outside the support.
    """
    bw_sq = bandwidth**2
    weights = np.maximum(bw_sq - distances**2, 0.0)
    return weights


def compute_bandwidth(coords: np.ndarray, min_neighbors: int = 35) -> float:
    """
    Compute the Epanechnikov kernel bandwidth parameter λ.

    For each sample location, find the distance to its (min_neighbors)-th
    nearest neighbor. The bandwidth is the maximum of these distances,
    ensuring every sample has at least `min_neighbors` within the kernel
    support.

    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
        Spatial coordinates (already scaled).
    min_neighbors : int, default 35
        Minimum number of neighbors within kernel support.

    Returns
    -------
    float
        Bandwidth parameter λ.
    """
    n = coords.shape[0]
    if min_neighbors >= n:
        warnings.warn(
            f"min_neighbors ({min_neighbors}) >= n ({n}). "
            f"Setting min_neighbors to n-1."
        )
        min_neighbors = n - 1

    tree = cKDTree(coords)
    # k+1 because query includes the point itself
    dists, _ = tree.query(coords, k=min_neighbors + 1)
    # Take the distance to the (min_neighbors)-th neighbor (last column)
    bandwidth = dists[:, -1].max()
    return bandwidth


def build_neighborhood_structure(
    coords: np.ndarray, bandwidth: float
) -> Tuple[list, list]:
    """
    Precompute neighborhoods and kernel weights for all sample locations.

    For each sample, find all other samples within the kernel bandwidth
    and compute their Epanechnikov kernel weights.

    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
        Spatial coordinates.
    bandwidth : float
        Kernel bandwidth λ.

    Returns
    -------
    neighbor_indices : list of np.ndarray
        neighbor_indices[t] = array of indices of samples within bandwidth of x_t.
    neighbor_weights : list of np.ndarray
        neighbor_weights[t] = corresponding Epanechnikov kernel weights.
    """
    tree = cKDTree(coords)
    n = coords.shape[0]

    neighbor_indices = []
    neighbor_weights = []

    for t in range(n):
        # Find all points within bandwidth
        idx = tree.query_ball_point(coords[t], r=bandwidth)
        idx = np.array(idx, dtype=int)

        # Compute distances and kernel weights
        dists = np.linalg.norm(coords[idx] - coords[t], axis=1)
        weights = epanechnikov_kernel(dists, bandwidth)

        neighbor_indices.append(idx)
        neighbor_weights.append(weights)

    return neighbor_indices, neighbor_weights


def compute_variogram_entry(
    t: int,
    t_prime: int,
    Z: np.ndarray,
    neighbor_indices: list,
    neighbor_weights: list,
) -> float:
    """
    Compute the sum of absolute estimated cross-variograms at a pair of
    sample locations (x_t, x_t'), i.e.:

        Σ_{i,j} |γ̂_{ij}(x_t, x_t')|

    This is the dissimilarity component before normalisation.

    The non-parametric kernel estimator (Equation 1 from the paper):

        γ̂_{ij}(u, v) = Σ_{l,l'} K(||u-x_l||) K(||v-x_l'||)
                        × (Z_i(x_l) - Z_i(x_l')) (Z_j(x_l) - Z_j(x_l'))
                        / (2 Σ_{l,l'} K(||u-x_l||) K(||v-x_l'||))

    For u ≠ v (the indicator function 1_{u≠v} = 1 when u ≠ v).

    Parameters
    ----------
    t : int
        Index of first sample location.
    t_prime : int
        Index of second sample location.
    Z : np.ndarray, shape (n, p)
        Standardised feature matrix.
    neighbor_indices : list of np.ndarray
        Precomputed neighborhood indices.
    neighbor_weights : list of np.ndarray
        Precomputed kernel weights.

    Returns
    -------
    float
        Sum of absolute cross-variogram estimates: Σ_{i,j} |γ̂_{ij}(x_t, x_t')|
    """
    p = Z.shape[1]

    # Neighbors and weights for x_t and x_t'
    idx_t = neighbor_indices[t]
    w_t = neighbor_weights[t]
    idx_tp = neighbor_indices[t_prime]
    w_tp = neighbor_weights[t_prime]

    n_t = len(idx_t)
    n_tp = len(idx_tp)

    if n_t == 0 or n_tp == 0:
        return 0.0

    # Weight matrix: outer product of kernel weights
    # W[l, l'] = K(||x_t - x_l||) * K(||x_t' - x_l'||)
    W = np.outer(w_t, w_tp)  # shape (n_t, n_tp)
    W_sum = W.sum()

    if W_sum < 1e-12:
        return 0.0

    # Feature values at neighbor locations
    Z_t = Z[idx_t]  # shape (n_t, p)
    Z_tp = Z[idx_tp]  # shape (n_tp, p)

    # Compute all cross-variogram entries simultaneously
    # For each variable pair (i, j):
    #   γ̂_{ij} = Σ_{l,l'} W[l,l'] (Z_i(x_l) - Z_i(x_l')) (Z_j(x_l) - Z_j(x_l'))
    #            / (2 * W_sum)
    #
    # Increments: ΔZ_i[l, l'] = Z_i(x_l) - Z_i(x_l')
    # We need: Σ_{l,l'} W[l,l'] * ΔZ_i[l,l'] * ΔZ_j[l,l']

    # Efficient computation using matrix operations:
    # For each variable i:
    #   weighted_sum_Z_i = Σ_l w_t[l] * Z_i(x_l)    (scalar per variable)
    #   weighted_sum_Z_i' = Σ_l' w_tp[l'] * Z_i(x_l')
    #
    # Σ_{l,l'} W[l,l'] * Z_i(x_l) * Z_j(x_l')
    #   = (Σ_l w_t[l] Z_i(x_l)) * (Σ_l' w_tp[l'] Z_j(x_l'))
    #   = a_i * b_j
    #
    # Expanding the increment product:
    # ΔZ_i * ΔZ_j = Z_i(l)Z_j(l') - Z_i(l)Z_j(l') - Z_i(l')Z_j(l) + Z_i(l')Z_j(l')
    #
    # Wait — this doesn't simplify as nicely. Let's be more careful.
    #
    # Σ W (Z_i(l) - Z_i(l'))(Z_j(l) - Z_j(l'))
    # = Σ W Z_i(l)Z_j(l) - Σ W Z_i(l)Z_j(l') - Σ W Z_i(l')Z_j(l) + Σ W Z_i(l')Z_j(l')
    #
    # Term 1: Σ_{l,l'} W[l,l'] Z_i(l) Z_j(l)
    #        = Σ_l (Σ_l' W[l,l']) Z_i(l) Z_j(l)
    #        = Σ_l row_sum[l] Z_i(l) Z_j(l)
    #
    # Term 2: Σ_{l,l'} W[l,l'] Z_i(l) Z_j(l')
    #        = (Σ_l w_t[l] Z_i(l)) (Σ_l' w_tp[l'] Z_j(l'))
    #        = a_i * b_j
    #
    # Term 3: = a_j * b_i  (symmetric swap)
    #
    # Term 4: Σ_{l,l'} W[l,l'] Z_i(l') Z_j(l')
    #        = Σ_l' (Σ_l W[l,l']) Z_i(l') Z_j(l')
    #        = Σ_l' col_sum[l'] Z_i(l') Z_j(l')

    # Row sums and column sums of weight matrix
    row_sums = W.sum(axis=1)  # shape (n_t,)    Σ_{l'} W[l, l']
    col_sums = W.sum(axis=0)  # shape (n_tp,)   Σ_{l}  W[l, l']

    # Weighted feature vectors
    a = w_t @ Z_t  # shape (p,)  a_i = Σ_l w_t[l] Z_i(x_l)
    b = w_tp @ Z_tp  # shape (p,)  b_j = Σ_l' w_tp[l'] Z_j(x_l')

    # Term 1: Σ_l row_sum[l] * Z(l) ⊗ Z(l)  → (p, p) matrix
    # = Z_t.T @ diag(row_sums) @ Z_t
    term1 = (Z_t * row_sums[:, None]).T @ Z_t  # (p, p)

    # Term 2: a ⊗ b → (p, p)
    term2 = np.outer(a, b)

    # Term 3: b ⊗ a → (p, p)  (but using a for first index, b for second)
    # Actually: term3 = Σ W Z_i(l') Z_j(l) = (Σ w_tp Z_i) (Σ w_t Z_j) = b_i * a_j
    term3 = np.outer(b, a)

    # Term 4: Σ_l' col_sum[l'] * Z(l') ⊗ Z(l')
    term4 = (Z_tp * col_sums[:, None]).T @ Z_tp  # (p, p)

    # Cross-variogram matrix (unnormalised by 2*W_sum)
    gamma_matrix = (term1 - term2 - term3 + term4) / (2.0 * W_sum)

    # Sum of absolute values
    return np.abs(gamma_matrix).sum()


def compute_similarity_matrix(
    coords: np.ndarray,
    Z: np.ndarray,
    min_neighbors: int = 35,
    scale_coords: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute the geostatistical similarity matrix.

    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
        Spatial coordinates. Will be min-max scaled to [0, 1] per dimension
        if scale_coords is True.
    Z : np.ndarray, shape (n, p)
        Feature variables. Should already be standardised (zero mean, unit
        variance) before calling this function.
    min_neighbors : int, default 35
        Minimum number of neighbors for bandwidth selection.
    scale_coords : bool, default True
        Whether to min-max scale coordinates to [0, 1] per dimension.
    verbose : bool, default True
        Print progress information.

    Returns
    -------
    S : np.ndarray, shape (n, n)
        Similarity matrix with values in [0, 1].
        S[t, t] = 1, S[t, t'] = S[t', t], S[t, t'] >= 0.
    """
    n = coords.shape[0]
    p = Z.shape[1]

    if verbose:
        print(f"Computing geostatistical similarity matrix")
        print(f"  n = {n} samples, p = {p} variables, d = {coords.shape[1]}D")

    # --- Step 0: Scale coordinates ---
    if scale_coords:
        coords_scaled = coords.copy().astype(float)
        for d in range(coords.shape[1]):
            col_min = coords_scaled[:, d].min()
            col_max = coords_scaled[:, d].max()
            col_range = col_max - col_min
            if col_range > 0:
                coords_scaled[:, d] = (coords_scaled[:, d] - col_min) / col_range
            else:
                coords_scaled[:, d] = 0.0
    else:
        coords_scaled = coords

    if verbose:
        print(f"  Coordinates scaled to [0, 1] per dimension")

    # --- Step 1: Compute bandwidth ---
    bandwidth = compute_bandwidth(coords_scaled, min_neighbors=min_neighbors)
    if verbose:
        print(f"  Bandwidth λ = {bandwidth:.6f}")

    # --- Step 2: Build neighborhoods ---
    if verbose:
        print(f"  Building neighborhood structure...")
    neighbor_indices, neighbor_weights = build_neighborhood_structure(
        coords_scaled, bandwidth
    )

    # Report neighborhood stats
    n_neighbors = [len(idx) for idx in neighbor_indices]
    if verbose:
        print(
            f"  Neighbors per sample: "
            f"min={min(n_neighbors)}, "
            f"median={int(np.median(n_neighbors))}, "
            f"max={max(n_neighbors)}"
        )

    # --- Step 3: Compute dissimilarity matrix (before normalisation) ---
    if verbose:
        print(f"  Computing pairwise variogram sums ({n*(n-1)//2} unique pairs)...")

    # D[t, t'] = Σ_{i,j} |γ̂_{ij}(x_t, x_t')|
    D = np.zeros((n, n))

    report_interval = max(1, n // 10)
    for t in range(n):
        if verbose and t % report_interval == 0:
            print(f"    Row {t}/{n} ({100*t/n:.0f}%)")

        for t_prime in range(t + 1, n):
            val = compute_variogram_entry(
                t, t_prime, Z, neighbor_indices, neighbor_weights
            )
            D[t, t_prime] = val
            D[t_prime, t] = val

    # --- Step 4: Normalise and convert to similarity ---
    G = D.max()
    if G > 0:
        S = 1.0 - D / G
    else:
        warnings.warn("All dissimilarity values are zero. Returning identity matrix.")
        S = np.ones((n, n))

    # Diagonal should be exactly 1
    np.fill_diagonal(S, 1.0)

    if verbose:
        print(f"  Normalising factor G = {G:.6f}")
        print(
            f"  Similarity range: [{S[np.triu_indices(n, k=1)].min():.4f}, "
            f"{S[np.triu_indices(n, k=1)].max():.4f}] (off-diagonal)"
        )
        print(f"  Done.")

    return S


# ---------------------------------------------------------------------------
# Convenience utilities
# ---------------------------------------------------------------------------


def standardise_features(Z_raw: np.ndarray) -> np.ndarray:
    """Z-score standardisation (zero mean, unit variance per column)."""
    mu = Z_raw.mean(axis=0)
    sigma = Z_raw.std(axis=0)
    sigma[sigma == 0] = 1.0  # avoid division by zero for constant columns
    return (Z_raw - mu) / sigma


def similarity_to_dissimilarity(S: np.ndarray) -> np.ndarray:
    """Convert similarity matrix to dissimilarity (for use with facility location solvers)."""
    return 1.0 - S


# ---------------------------------------------------------------------------
# Example / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data: 200 samples in 2D with 3 features
    n = 200
    coords = np.random.rand(n, 2) * 1000  # 1000m x 1000m area
    # Create spatially correlated features (simple demo)
    Z_raw = np.column_stack(
        [
            np.sin(coords[:, 0] / 200) + np.random.randn(n) * 0.3,
            np.cos(coords[:, 1] / 150) + np.random.randn(n) * 0.3,
            coords[:, 0] / 500 + np.random.randn(n) * 0.2,
        ]
    )

    Z = standardise_features(Z_raw)

    print("=" * 60)
    print("Geostatistical Similarity Matrix - Synthetic Example")
    print("=" * 60)

    S = compute_similarity_matrix(coords, Z, min_neighbors=20, verbose=True)

    print(f"\nSimilarity matrix shape: {S.shape}")
    print(f"Sample S[0:5, 0:5]:")
    print(np.round(S[:5, :5], 3))

    # Convert to dissimilarity for downstream use
    D = similarity_to_dissimilarity(S)
    print(f"\nDissimilarity matrix ready for facility location solvers.")
    print(f"D shape: {D.shape}, range: [{D.min():.4f}, {D.max():.4f}]")
