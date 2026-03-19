import numpy as np
from pandas import concat
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus
from scipy.spatial.distance import cdist


def kmeans_representative_sampling(
    X: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    return_centroids: bool = False,
):
    """
    Pick `k` representative points by taking the member of each k‑means cluster
    that is closest to its centroid (i.e. a medoid).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input features.
    k : int
        Desired number of representatives (== number of clusters).
    seed : int, default 0
        RNG seed passed to MiniBatchKMeans for reproducibility.
    normalize : bool, default False
        If True, z‑score the features before clustering.
    return_centroids : bool, default False
        If True, also return the centroid coordinates.

    Returns
    -------
    indices : ndarray, shape (k,)
        Row positions of the selected representatives (0‑based like .iloc).
    centroids : ndarray, shape (k, n_features), optional
        Returned only if `return_centroids` is True.
    """
    n = X.shape[0]
    if n <= k:
        # nothing to compress
        out = np.arange(n)
        return (out, X.copy()) if return_centroids else out

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=1024,
        n_init="auto",
    ).fit(X)

    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    closest = closest.astype(int)

    if return_centroids:
        return closest, km.cluster_centers_
    return closest


def kmeans_representative_sampling_with_locked(
    X: np.ndarray,
    k: int,
    locked_indices: np.ndarray,
    *,
    seed: int = 0,
    return_centroids: bool = False,
):
    """
    Pick `k` NEW representative points via k-means while honoring a fixed set of
    legacy (already-sampled) indices.

    Legacy indices are ADDITIONAL to `k` — if you have 5 legacy samples and want
    7 new ones, pass k=7 and expect 12 indices back.

    Strategy
    --------
    1. Seed k_total = k + len(locked) cluster centers: legacy positions first,
       then kmeans++ initialization for the k free clusters.
    2. Fit MiniBatchKMeans with that initialization (n_init=1 — the seeding is
       intentional, not a starting guess to be improved by multi-restart).
    3. Find medoids for all k_total clusters, restricting eligible candidates to
       non-legacy points so legacy indices can never "steal" a free slot.
    4. Union free medoids with locked_indices and deduplicate.

    Constraint model: **soft, not hard.**
    sklearn's MiniBatchKMeans moves all cluster centers during fitting — there is
    no native freeze mechanism.  The initialization biases clusters to form in
    non-legacy space; the medoid-exclusion step (step 3) is the hard guarantee
    that legacy indices appear in the output.  If you need truly hard-locked
    cluster centers, a custom EM loop would be required.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix — should already be ILR-transformed and StandardScaler-
        normalized, same as `kmeans_representative_sampling`.
    k : int
        Number of NEW (free) representatives to select.  Legacy samples are
        additional to this count.
    locked_indices : array-like of int
        Positional (0-based, iloc-compatible) indices of legacy samples that
        must appear in the output.  Duplicates are silently removed.
    seed : int, default 0
        RNG seed for kmeans_plusplus initialization of the free clusters.
    return_centroids : bool, default False
        If True, also return the fitted cluster center coordinates.

    Returns
    -------
    indices : ndarray, shape (k + len(unique_locked),)
        Positional indices of selected points — legacy + new medoids, sorted.
        May be slightly shorter than k + len(locked) if multiple free clusters
        collapse onto the same eligible medoid; downstream `kcenter_greedy` with
        already_selected=indices will fill any such gap.
    centroids : ndarray, shape (k_total, n_features), optional
        Returned only if `return_centroids` is True.

    Known edge cases
    ----------------
    - If locked indices are tightly clustered in one region, many of the k_total
      clusters will form there and the k free clusters may cover less ground.
      Consider whether legacy samples are spatially representative before using
      this function.
    - If k == 0, returns locked_indices immediately.
    - If n_samples <= k + len(locked), returns all row indices.
    """
    locked = np.unique(np.asarray(locked_indices, dtype=int))
    k_locked = len(locked)

    if k == 0:
        return (locked, X[locked]) if return_centroids else locked

    n = X.shape[0]
    k_total = k + k_locked
    if n <= k_total:
        out = np.arange(n)
        return (out, X.copy()) if return_centroids else out

    # Build a (k_total, n_features) init array: legacy positions first, then
    # kmeans++ picks for the k free clusters.  Legacy positions anchor clusters
    # in already-covered space so the free clusters are pushed elsewhere.
    locked_centers = X[locked]  # (k_locked, n_features)
    free_init, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)
    full_init = np.vstack([locked_centers, free_init])  # (k_total, n_features)

    km = MiniBatchKMeans(
        n_clusters=k_total,
        init=full_init,
        n_init=1,  # initialization is intentional — no multi-restart
        random_state=seed,
        batch_size=1024,
    ).fit(X)

    # Restrict medoid candidates to non-legacy points so legacy indices are
    # never "used up" as free medoids.
    eligible_mask = np.ones(n, dtype=bool)
    eligible_mask[locked] = False
    eligible_idx = np.where(eligible_mask)[0]  # (n - k_locked,)

    closest_in_eligible, _ = pairwise_distances_argmin_min(
        km.cluster_centers_, X[eligible_idx]
    )
    free_medoids = eligible_idx[closest_in_eligible]  # map back to global idx

    all_indices = np.unique(np.concatenate([locked, free_medoids]))

    if return_centroids:
        return all_indices, km.cluster_centers_
    return all_indices


def kcenter_greedy(
    X,
    k: int,
    metric: str = "euclidean",
    *,
    seed: int = 0,
    already_selected=None,
    start_idx=None,
    return_dist: bool = False,
):
    """
    Greedy 2‑approximation to the k‑center problem with optional fixed centers.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data matrix (should already be scaled if you use Euclidean distance).
    k : int
        Desired total number of centers (includes any already_selected).
    metric : str, default 'euclidean'
        Any metric understood by scipy.spatial.distance.cdist.
    seed : int, default 0
        RNG seed used *only* when start_idx is randomly chosen.
    already_selected : 1‑D iterable of int, optional
        Indices that must be included in the final coreset.
        They count toward `k`.  Duplicates are ignored.
    start_idx : int, optional
        If supplied and `already_selected` is None/empty, this point is taken
        as the first center instead of choosing at random.
    return_dist : bool, default False
        If True, also return the final nearest‑center distances.

    Returns
    -------
    centers : ndarray, shape (k,)
        Positional indices of the selected centers, in selection order.
    dists : ndarray, shape (n_samples,), optional
        Distance from each point in X to its nearest selected center.
        Returned only if `return_dist` is True.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # ----- normalise input ---------------------------------------------------
    if already_selected is None:
        already_selected = []
    centers = np.unique(np.asarray(already_selected, dtype=int)).tolist()

    if len(centers) > k:
        raise ValueError(f"`already_selected` contains {len(centers)} indices ≥ k={k}.")

    # ----- choose seed if none provided --------------------------------------
    if len(centers) == 0:
        if start_idx is None:
            start_idx = rng.integers(n)
        centers.append(int(start_idx))

    # ----- initialise nearest‑center distance vector -------------------------
    # Stack all current centers once, compute distances in one go, take min.
    dists = cdist(X, X[centers], metric=metric).min(axis=1)
    dists[centers] = 0.0  # exact zeros for the chosen points

    # ----- greedy additions ---------------------------------------------------
    while len(centers) < k:
        nxt = int(dists.argmax())  # farthest point
        centers.append(nxt)

        # update distance vector in one shot
        dists = np.minimum(dists, cdist(X, X[[nxt]], metric=metric).ravel())

    centers = np.asarray(centers)

    if return_dist:
        return centers, dists
    return centers


# ----------------------------------------------------------------------------
# LEGACY SAMPLING METHODS (kept for reference)
# ----------------------------------------------------------------------------
def k_furthest_sampling(
    df, cols_data, n_samples=10, k_neighbors=5, random_seed=42, normalize=False
):
    """
    K-furthest sampling: selects points that are farthest from each other.

    Parameters:
        X (np.ndarray): Feature matrix of shape (n_points, n_features)
        n_samples (int): Number of samples to select
        k_neighbors (int): Used to estimate local density (higher = smoother density)
        random_seed (int): Random seed for reproducibility
        normalize (bool): Whether to normalize the data before sampling

    Returns:
        selected_indices (list): Indices of selected samples
    """
    np.random.seed(random_seed)
    X = df[cols_data].to_numpy()
    if normalize:
        X = StandardScaler().fit_transform(X)

    n_points = X.shape[0]
    if n_samples > n_points:
        raise ValueError("n_samples cannot be greater than number of points in X")

    # Step 1: Estimate local density (mean distance to k neighbors)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    local_density = distances[:, 1:].mean(axis=1)  # skip self-distance at index 0

    # Step 2: K-center selection
    selected_indices = []
    remaining_indices = np.arange(n_points)

    # Initialize with a random point
    start_idx = np.random.choice(remaining_indices)
    selected_indices.append(start_idx)
    remaining_indices = np.delete(
        remaining_indices, np.where(remaining_indices == start_idx)
    )

    for _ in range(1, n_samples):
        # Compute distance from each unselected point to nearest selected point
        dists_to_selected = pairwise_distances(
            X[remaining_indices], X[selected_indices]
        )
        min_dists = dists_to_selected.min(axis=1)

        # Select point with maximum distance to any selected point
        best_idx = remaining_indices[np.argmax(min_dists)]
        selected_indices.append(best_idx)
        remaining_indices = np.delete(
            remaining_indices, np.where(remaining_indices == best_idx)
        )

    return df.iloc[selected_indices]


def density_weighted_sampling(
    df, cols, n_samples=300, k=20, random_seed=0, normalize=False
):
    X = df[cols].to_numpy()
    if normalize:
        X = StandardScaler().fit_transform(X)

    # Local density ≈ 1 / mean k‑NN distance
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nn.kneighbors(X)
    density = 1 / (dists[:, 1:].mean(axis=1) + 1e-9)

    weights = 1 / density  # inverse density
    weights /= weights.sum()

    rng = np.random.default_rng(random_seed)
    sel = rng.choice(len(df), size=n_samples, replace=False, p=weights)
    return df.iloc[sel]


def kpp_sampling(df, cols, n_samples=300, random_seed=0, normalize=False):
    X = df[cols].to_numpy()
    if normalize:
        X = StandardScaler().fit_transform(X)

    centers, idx = kmeans_plusplus(
        X, n_clusters=n_samples, random_state=random_seed, n_local_trials=None
    )
    return df.iloc[idx]


def kcenter_sampling(df, cols, n_samples=300, random_seed=0, normalize=False):
    X = df[cols].to_numpy()
    if normalize:
        X = StandardScaler().fit_transform(X)

    rng = np.random.default_rng(random_seed)
    sel = [rng.integers(len(X))]  # initial seed
    dist = pairwise_distances(X, X[sel]).ravel()

    for _ in range(n_samples - 1):
        idx = dist.argmax()
        sel.append(idx)
        dist = np.minimum(dist, pairwise_distances(X, X[[idx]]).ravel())

    return df.iloc[sel]


def kmeans_representative_sampling_legacy(
    df, cols_data, random_seed, n_samples=10, normalize=False
):
    """
    Selects representative samples using k-means centroids.

    Parameters:
        df (pd.DataFrame): Input data
        cols_data (list): List of columns to use for clustering
        random_seed (int): Random seed
        n_samples (int): Number of samples to return
        normalize (bool): Whether to normalize features

    Returns:
        pd.DataFrame: Subset of df with selected samples (iloc-based)
    """
    X = df[cols_data].to_numpy()
    if normalize:
        X = StandardScaler().fit_transform(X)

    if len(df) <= n_samples:
        return df.copy()

    kmeans = MiniBatchKMeans(
        n_clusters=n_samples, random_state=random_seed, batch_size=1048, n_init="auto"
    ).fit(X)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

    return df.iloc[closest]


def cluster_entropy(X_cluster, k=5):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_cluster)
    dists, _ = nbrs.kneighbors(X_cluster)
    return np.mean(np.log(dists[:, 1:].mean(axis=1)))


def add_tail_samples(df_full, df_sampled, col, random_seed, pct=0.02, n_each=5):
    df_full = df_full.copy()
    df_full["log_" + col] = np.log1p(df_full[col])

    lower_cut = df_full["log_" + col].quantile(pct)
    upper_cut = df_full["log_" + col].quantile(1 - pct)

    tail_candidates = df_full[
        (df_full["log_" + col] < lower_cut) | (df_full["log_" + col] > upper_cut)
    ]
    tail_candidates = tail_candidates.loc[~tail_candidates.index.isin(df_sampled.index)]

    # Sample from tails (balanced high/low if enough data)
    lower = tail_candidates[tail_candidates["log_" + col] < lower_cut]
    upper = tail_candidates[tail_candidates["log_" + col] > upper_cut]
    n = min(n_each, len(lower), len(upper))

    return concat(
        [
            df_sampled,
            lower.sample(n=n, random_state=random_seed),
            upper.sample(n=n, random_state=random_seed),
        ]
    )


def calc_curr_count_q(srs, idx_sampled):
    """
    Calculate the current proportion of samples in the specified quantile.
    """
    return srs.loc[idx_sampled].value_counts()


def balance_distribution(n_samples_to_take, srs_q_cnt, pre_alloc):
    # calculate the number samples
    n_sampled = srs_q_cnt.sum()
    # get number of bins
    n_bins = len(srs_q_cnt)
    # calculate the target n_per_bin of this step
    # this will be the number we have, plus the number we need
    # distributed accross the total number of bins
    target = (n_sampled + n_samples_to_take) / n_bins
    # calculate how many per bin
    # this is the max of 0 and the number already sampled-target
    srs_needed = (target - srs_q_cnt).clip(lower=0)
    total_needed = srs_needed.sum()

    if total_needed == 0:
        srs_needed = pre_alloc.copy()
    # if we're below or equal distribute evenly
    elif total_needed <= n_samples_to_take:
        n_additional = n_samples_to_take - total_needed
        per_group = n_additional // n_bins
        n_extra = int(n_additional % n_bins)
        srs_needed += per_group
        for idx in range(n_extra):
            srs_needed.iloc[idx] += 1
    else:
        # Scale down proportionally to match n_samples_to_take
        scaled = srs_needed / total_needed * n_samples_to_take
        floored = scaled.astype(int)
        remainder = n_samples_to_take - floored.sum()

        # Greedily add remainder to highest fractional values
        fractional = scaled - floored
        for idx in fractional.sort_values(ascending=False).index[:remainder]:
            floored[idx] += 1

        srs_needed = floored

    return srs_needed.astype(int)
