import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

RANDOMSTATE = 42  # For reproducibility


def plot_ecdf(df_full, df_subset, col, log=False, normalize=False, **kwargs):
    """
    Plot the empirical cumulative distribution function (ECDF) of a column
    from two datasets: a sampled subset and the full dataset.
    Parameters
    ----------
    df_full: full dataset (pandas DataFrame) - typically the full dataset
    df_subset: sampled dataset (pandas DataFrame) - typically the subset dataset
    col: column name to plot ECDF for
    log: whether to apply log transformation to the column values before plotting
    normalize: whether to normalize the column values before plotting
    Returns
    -------
    fig, ax: matplotlib figure and axis objects containing the ECDF plot
    """
    _kwargs = {
        "c_subset": "blue",
        "c_full": "orange",
        "label_subset": "SAMPLED",
        "label_full": "FULL",
    }
    _kwargs.update(kwargs)
    if len(df_subset) > len(df_full):
        raise ValueError(
            "Subset dataset cannot be larger than full dataset for ECDF comparison."
        )

    fnc_apply = np.log if log else lambda x: x
    srs_subset = df_subset[col].apply(fnc_apply)
    srs_full = df_full[col].apply(fnc_apply)
    if normalize:
        scaler = StandardScaler()
        srs_subset = scaler.fit_transform(
            srs_subset.to_numpy().reshape(-1, 1)
        ).flatten()
        srs_full = scaler.transform(srs_full.to_numpy().reshape(-1, 1)).flatten()
    else:
        srs_subset = srs_subset.to_numpy()
        srs_full = srs_full.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.ecdfplot(
        srs_subset,
        label=_kwargs["label_subset"],
        color=_kwargs["c_subset"],
        ax=ax,
    )
    sns.ecdfplot(
        srs_full,
        label=_kwargs["label_full"],
        color=_kwargs["c_full"],
        ax=ax,
    )
    st_title = "ECDF of " + col
    if normalize:
        st_title += " (normalized)"
    if log:
        st_title += " (log transformed)"
    ax.set_title(st_title)
    ax.set_xlabel(col)
    ax.set_ylabel("ECDF")
    ax.legend()
    return fig, ax


def calc_ks_wasserstein(df_full, df_subset, col, log=False, normalize=False):
    """
    Calculate the Kolmogorov-Smirnov statistic and Wasserstein distance
    between two datasets for a given column.
    Parameters
    ----------
    df_subset: sampled dataset (pandas DataFrame) - typically the subset dataset
    df_full: full dataset (pandas DataFrame) - typically the full dataset
    col: column name to compare
    log: whether to apply log transformation to the column values before comparison
    normalize: whether to normalize the column values before comparison
    Returns
    -------
    ks_stat: Kolmogorov-Smirnov statistic
    ks_p: p-value from the Kolmogorov-Smirnov test
    w_dist: Wasserstein distance between the two distributions

    """
    if len(df_subset) > len(df_full):
        raise ValueError(
            "Subset dataset cannot be larger than full dataset for KS/Wasserstein comparison."
        )
    fnc_apply = np.log if log else lambda x: x
    srs_subset = df_subset[col].apply(fnc_apply)
    srs_full = df_full[col].apply(fnc_apply)
    # normalize to ensure same scale
    if normalize:
        scaler = StandardScaler()
        srs_subset = scaler.fit_transform(
            srs_subset.to_numpy().reshape(-1, 1)
        ).flatten()
        srs_full = scaler.transform(srs_full.to_numpy().reshape(-1, 1)).flatten()
    else:
        srs_subset = srs_subset.to_numpy()
        srs_full = srs_full.to_numpy()

    # Calculate KS statistic and p-value
    ks_stat, ks_p = ks_2samp(srs_subset, srs_full)
    # Calculate Wasserstein distance
    w_dist = wasserstein_distance(srs_subset, srs_full)

    return ks_stat, ks_p, w_dist


def pca_reconstruction_error(
    X_full,
    X_sample,
    n_components=None,
    scale=True,
    random_state=RANDOMSTATE,
    verbose=True,
):
    """
    Calculate PCA reconstruction error between a full dataset and a sampled subset.
    Parameters
    ----------
    X_full: np.ndarray - full dataset (2D array)
    X_sample: np.ndarray - sampled subset (2D array)
    n_components: int, optional - number of PCA components to use; if None, defaults to min of both datasets' features
    scale: bool, optional - whether to standardize features before PCA; default is True
    random_state: int, optional - random state for reproducibility; default is 42
    verbose: bool, optional - whether to print progress messages; default is True
    Returns
    -------
    mse: float - mean squared error of the reconstruction
    explained: float - total variance explained by the PCA components
    """
    if scale:
        scaler = StandardScaler()
        X_sample = scaler.fit_transform(X_sample)
        X_full = scaler.transform(X_full)
    if n_components is None:
        n_components = min(X_sample.shape[1], X_full.shape[1])

    # Fit PCA on sampled subset
    if verbose:
        print(
            f"Seeding PCA with random state {random_state} and {n_components} components"
        )
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_sample)

    # Project full data into sample-trained PCA space, then reconstruct
    X_full_transformed = pca.transform(X_full)
    X_reconstructed = pca.inverse_transform(X_full_transformed)

    # Compare reconstructed to original full data
    mse = mean_squared_error(X_full, X_reconstructed)

    # Optional: variance explained
    explained = np.sum(pca.explained_variance_ratio_)

    return mse, explained


def cluster_variance_coverage(X_full, X_sampled):
    """
    Calculate the variance coverage of a sampled dataset compared to the full dataset.
    Parameters
    ----------
    X_full: np.ndarray - full dataset (2D array)
    X_sampled: np.ndarray - sampled subset (2D array)
    Returns
    -------
    coverage: float - ratio of variance in sampled data to variance in full data
    """
    return np.var(X_sampled, axis=0).sum() / np.var(X_full, axis=0).sum()


from scipy.spatial.distance import cdist


def calculate_max_radius(X_full_std, X_sampled_std):
    """
    Calculate the maximum radius of the sampled dataset compared to the full dataset.
    Parameters
    ----------
    X_full_std: np.ndarray - standardized full dataset (2D array)
    X_sampled_std: np.ndarray - standardized sampled subset (2D array)
    Returns
    -------
    max_radius: float - maximum radius of the sampled data points from the origin
    """
    nearest = cdist(X_full_std, X_sampled_std, metric="euclidean").min(axis=1)
    max_radius = nearest.max()
    return max_radius
