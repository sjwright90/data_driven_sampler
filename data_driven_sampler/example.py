# %%
# example script for using the data-driven sampling package
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from src.ilr_functions import calculate_sbp_from_hca, make_basis_formula
from src.data_samplers import kmeans_representative_sampling, kcenter_greedy
from src.metric_functions import plot_ecdf

# from scipy.stats import energy_distance
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity

from sklearn.feature_selection import mutual_info_regression


from sklearn.preprocessing import StandardScaler


# MiniBatchKmeans throws a memory leak warning
# but with larger datasets it is not an issue
# if you're concerned uncomment the max cores setting below
import os

os.environ["OMP_NUM_THREADS"] = "4"
# calculate custom with ceil(n_samples / 1024) to avoid memory issues

# alternatively suppress warnings
# import warnings


# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# --------------------------------------------------------------------------
# DEFINE FUNCTIONS
# --------------------------------------------------------------------------
def calculate_energy_distance(X, Y):
    """
    Calculate the energy distance between two datasets X and Y.
    Parameters
    ----------
    X: numpy array of shape (n_samples_X, n_features)
        First dataset.
    Y: numpy array of shape (n_samples_Y, n_features)
        Second dataset.
    Returns
    -------
    float
        The energy distance between datasets X and Y.
    """
    d_xy = cdist(X, Y, metric="euclidean")
    d_xx = cdist(X, X, metric="euclidean")
    d_yy = cdist(Y, Y, metric="euclidean")

    term1 = 2 * d_xy.mean()
    term2 = d_xx.mean()
    term3 = d_yy.mean()

    energy_dist = term1 - term2 - term3
    return energy_dist


# --------------------------------------------------------------------
# GENERATE DATA
# --------------------------------------------------------------------
# %% generate some example data
RANDOMSTATE = 42
# set random seed for reproducibility
np.random.seed(RANDOMSTATE)

n_observations = 1000  # number of observations to generate
n_samples = 85  # number of samples to take
n_features = 10  # number of features to generate
n_proportion_outliers = 0.2  # proportion of outliers to generate

rng = np.random.default_rng(RANDOMSTATE)
core = rng.normal(
    scale=1.2, size=(int(n_observations * (1 - n_proportion_outliers)), n_features)
)
outliers = rng.uniform(
    -11, 11, size=(int(n_observations * n_proportion_outliers), n_features)
)
X = np.vstack([core, outliers])
# need to shift all so positive, >0
X += np.abs(X.min()) + 1
# %%
# --------------------------------------------------------------------
# CALCULATE ILR BASIS
# --------------------------------------------------------------------
# put into a DataFrame since 'calculate_sbp_from_hca' expects a DataFrame
df_X = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
df_X["EXAMPLE_CATEGORICAL"] = np.random.choice(
    ["A", "B", "C"], size=df_X.shape[0], p=[0.3, 0.5, 0.2]
)

df_sbp = df_X.drop(columns="EXAMPLE_CATEGORICAL")
# do any log or scaling to the df_sbp if needed
# not shown here
ilr_embedding = calculate_sbp_from_hca(
    df_sbp.T
)  # make sure analytes/elements are on rows

# make new df to put ilr into (or just use df_X)
df_ilr = df_X[["EXAMPLE_CATEGORICAL"]].copy()
for idx, v in ilr_embedding.items():
    # use ORIGINAL (i.e. untransformed) df_X for the basis calculation
    df_ilr[f"ILR_{idx}"] = make_basis_formula(df_X, v["Numerator"], v["Denominator"])
# you now have a DataFrame with ilr features, you can use it for sampling
# alternatively, transform with CLR
# %%
# quick view of the ilr embedding
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df_ilr,
    x="ILR_0",
    y="ILR_2",
    hue="EXAMPLE_CATEGORICAL",
    style="EXAMPLE_CATEGORICAL",
    ax=ax,
)
# %%
# --------------------------------------------------------------------
# RUN SAMPLER
# --------------------------------------------------------------------

# to estimate best ratio of centroids to outliers
# we will step across a range of proportions
# then plot the maximin distnce to the sampled
# points, using and "elbow" to determine the best ratio
n_steps = 20  # number of steps to take, 20 gives good resolution
maximin_distance_per_step = np.zeros((n_steps,))  # array to hold maximin distances
linspace_ratios = np.linspace(0.1, 1.0, n_steps)  # range of proportions
X_ilr = df_ilr.drop(
    columns="EXAMPLE_CATEGORICAL"
).to_numpy()  # our ilr data as numpy array

# scale if needed
scaler = StandardScaler()
X_ilr = scaler.fit_transform(X_ilr)

# calculate baseline for full data
vol_full = ConvexHull(X_ilr).volume
kde_full = KernelDensity(bandwidth=0.5).fit(X_ilr)
# %%
for idx, prop in enumerate(linspace_ratios):
    n_centroids = max(1, int(n_samples * prop))  # number of centroids to sample

    # I am seriously considering swapping this out for
    # a random or stratified random sampling
    idx_centroids_kmeans = kmeans_representative_sampling(
        X=X_ilr,
        k=n_centroids,
        seed=RANDOMSTATE,
    )
    # sample remainder with kcenter_greedy
    # pass the full n_samples and the already selected centroids
    # the function expects the total number of samples
    idx_centroids_kmeans_greedy = kcenter_greedy(
        X=X_ilr,
        k=n_samples,  # total samples
        metric="euclidean",
        already_selected=idx_centroids_kmeans,  # already selected centroids
        seed=RANDOMSTATE,  # actually ignored in this case
    )
    X_sampled_cent_greedy = X_ilr[idx_centroids_kmeans_greedy, :]
    arry_cent_greedy = cdist(X_ilr, X_sampled_cent_greedy, metric="euclidean").min(
        axis=1
    )
    maximin_distance_per_step[idx] = arry_cent_greedy.max()
# %%
# then plot the maximin distances
# this one does not make much sense b/c generated data is uniform
# but you'll get the idea
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    linspace_ratios,
    maximin_distance_per_step,
    marker="o",
    linestyle="-",
    color="blue",
)
_ = ax.grid()
_ = ax.set_xlabel("Proportion of samples used as centroids")
_ = ax.set_ylabel("Max distance to centroids (σ-units)")
_ = ax.set_title(
    f"Max distance to centroids vs Proportion of samples used as centroids\n{n_samples} samples"
)
# %%
# --------------------------------------------------------------------
# RUN SAMPLER WITH BEST PROPORTION
# --------------------------------------------------------------------
optimal_proportion = 0.6  # this is just an example, use the elbow from the plot above
n_centroids = max(
    1, int(n_samples * optimal_proportion)
)  # number of centroids to sample

# again this might be better with random or stratified random sampler
idx_centroids_kmeans_final = kmeans_representative_sampling(
    X=X_ilr,
    k=n_centroids,
    seed=RANDOMSTATE,
)

idx_centroids_kmeans_greedy_final = kcenter_greedy(
    X=X_ilr,
    k=n_samples,  # total samples
    metric="euclidean",
    already_selected=idx_centroids_kmeans_final,  # already selected centroids
    seed=RANDOMSTATE,  # actually ignored in this case
)
X_sampled_cent_greedy_final = X_ilr[idx_centroids_kmeans_greedy_final, :]
df_sampled_ilr = df_ilr.iloc[
    idx_centroids_kmeans_greedy_final, :
].copy()  # Row indices for .iloc (not .loc)
# you might need a different indexer for the raw data
# a sample name column would be ideal
df_sampled_raw = df_X.iloc[
    idx_centroids_kmeans_greedy_final, :
].copy()  # Row indices for .iloc (not .loc)
# %%
# calculate some metrics on the sampled data
mean_distance_final = np.mean(
    cdist(X_ilr, X_sampled_cent_greedy_final, metric="euclidean").min(axis=1)
)
maximin_distance_final = (
    cdist(X_ilr, X_sampled_cent_greedy_final, metric="euclidean").min(axis=1).max()
)
coverage_radius_final = np.max(
    cdist(X_ilr, X_sampled_cent_greedy_final, metric="euclidean").min(axis=1)
)
hull_volume_ratio_final = ConvexHull(X_sampled_cent_greedy_final).volume / vol_full
kde_coverage_final = kde_full.score_samples(X_sampled_cent_greedy_final).mean()
energy_distance_final = calculate_energy_distance(X_ilr, X_sampled_cent_greedy_final)
# %%
# --------------------------------------------------------
# RUN SOME SIMPLE METRICS
# --------------------------------------------------------
# plot ECDF of the sampled vs full dataset
fig, ax = plot_ecdf(
    df_full=df_ilr,
    df_subset=df_sampled_ilr,
    col="ILR_4",  # change to any ILR column you want to compare
    log=False,  # optional
    normalize=False,  # optional, fits on the subset and transforms the full dataset
)
# actually does not do well with random, uniform data
# %%
# --------------------------------------------------------
# COMPARE TO RANDOM SAMPLING
# --------------------------------------------------------
n_random_draws = 1000
n_samples_random = 85
maximin_distance_random = np.zeros((n_random_draws,))
mean_distance_random = np.zeros((n_random_draws,))
coverage_radius_random = np.zeros((n_random_draws,))
hull_volume_ratio_random = np.zeros((n_random_draws,))
kde_coverage_random = np.zeros((n_random_draws,))
energy_distance_random = np.zeros((n_random_draws,))
for i in range(n_random_draws):
    np.random.seed(RANDOMSTATE + i)  # different seed each iteration
    idx_random = np.random.choice(X_ilr.shape[0], n_samples_random, replace=False)
    X_sampled_random = X_ilr[idx_random, :]
    arry_random = cdist(X_ilr, X_sampled_random, metric="euclidean").min(axis=1)
    maximin_distance_random[i] = arry_random.max()
    mean_distance_random[i] = arry_random.mean()
    coverage_radius_random[i] = np.max(arry_random)
    hull_volume_ratio_random[i] = ConvexHull(X_sampled_random).volume / vol_full
    kde_coverage_random[i] = kde_full.score_samples(X_sampled_random).mean()
    energy_distance_random[i] = calculate_energy_distance(X_ilr, X_sampled_random)
    if (i + 1) % 100 == 0:
        print(f"Completed {i+1} of {n_random_draws} random draws")
# %%
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(
    maximin_distance_random,
    bins=30,
    kde=True,
    color="gray",
    stat="density",
    label="Random Sampling",
    ax=ax,
)
ax.axvline(
    maximin_distance_final,
    color="blue",
    linestyle="--",
    label="KMeans + KCenter-Greedy",
)
_ = ax.set_xlabel("Max distance to sampled points (σ-units)")
_ = ax.set_ylabel("Density")
_ = ax.set_title(
    f"Maximin distance to sampled points\n{n_samples_random} samples, {n_random_draws} random draws"
)
_ = ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
# random gives us a good approximation of the distribution
# but kmeans+kcenter_greedy covers the space better
# again I might make an argument for using random or stratified random sampling
# for the first step then k-greedy for the second step
# but that's an evolving conversation with Morgan and others
# %%
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(
    mean_distance_random,
    bins=30,
    kde=True,
    color="gray",
    stat="density",
    label="Random Sampling",
    ax=ax,
)
ax.axvline(
    mean_distance_final,
    color="blue",
    linestyle="--",
    label="KMeans + KCenter-Greedy",
)

_ = ax.set_xlabel("Mean distance to sampled points (σ-units)")
_ = ax.set_ylabel("Density")
_ = ax.set_title(
    f"Mean distance to sampled points\n{n_samples_random} samples, {n_random_draws} random draws\nSmaller is better"
)
_ = ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
# %%
# Coverage Radius
# Radius needed to ensure all points are within distance R of a sample
# Related to maximin but gives intuitive "coverage guarantee"

fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(
    coverage_radius_random,
    bins=30,
    kde=True,
    color="gray",
    stat="density",
    label="Random Sampling",
    ax=ax,
)
ax.axvline(
    coverage_radius_final,
    color="blue",
    linestyle="--",
    label="KMeans + KCenter-Greedy",
)
_ = ax.set_xlabel("Coverage Radius (σ-units)")
_ = ax.set_ylabel("Density")
_ = ax.set_title(
    f"Coverage Radius\n{n_samples_random} samples, {n_random_draws} random draws\nSmaller is better"
)
_ = ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

# %%
# convex hull radius
# Radius of the convex hull around sampled points

fig, ax = plt.subplots(figsize=(8, 6))
_ = ax.set_xscale("log")

sns.histplot(
    hull_volume_ratio_random,
    bins=30,
    kde=True,
    color="gray",
    stat="density",
    label="Random Sampling",
    ax=ax,
)
ax.axvline(
    hull_volume_ratio_final,
    color="blue",
    linestyle="--",
    label="KMeans + KCenter-Greedy",
)
_ = ax.set_xlabel("Convex Hull Volume Ratio (Sampled / Full)")
_ = ax.set_ylabel("Density")
_ = ax.set_title(
    f"Convex Hull Volume Ratio (Sampled / Full)\n{n_samples_random} samples, {n_random_draws} random draws\nPerfect = 1.0"
)
_ = ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
# %%
# KDE Coverage
# Average log-likelihood of sampled points under the full data KDE
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(
    kde_coverage_random,
    bins=30,
    kde=True,
    color="gray",
    stat="density",
    label="Random Sampling",
    ax=ax,
)
ax.axvline(
    kde_coverage_final,
    color="blue",
    linestyle="--",
    label="KMeans + KCenter-Greedy",
)
_ = ax.set_xlabel("Average Log-Likelihood under Full Data KDE")
_ = ax.set_ylabel("Density")
_ = ax.set_title(
    f"Average Log-Likelihood under Full Data KDE\n{n_samples_random} samples, {n_random_draws} random draws\nHigher is better"
)
_ = ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
# actually underperforms random sampling here
# but that's by design since we are covering extremes
# %%

# %%
