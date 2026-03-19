# CLAUDE.md — data_driven_sampler

## Project Overview

R&D package for **distribution-based sampling** of high-dimensional datasets, with emphasis on compositional data (geochemistry, proportional features) and spatial/geostatistical contexts. The core problem: select a representative subset of `k` points from `n` that preserves distributional, geometric, and spatial properties.

**Domain**: Geoscience / geochemistry data analysis (Life Cycle Geo, LLC)

---

## Architecture

```
data_driven_sampler/
└── src/
    ├── data_samplers.py     # Sampling algorithms (k-means medoids, k-center greedy, legacy methods)
    ├── ilr_functions.py     # Isometric Log-Ratio (ILR) transforms for compositional data
    └── metric_functions.py  # Sample quality metrics and ECDF visualization
geostats_cluster.py          # Spatial similarity matrix (Fouedjio 2016/2017) — not yet packaged
```

**Primary workflow**: Raw data → ILR transform → KMeans centroids → KCenter-greedy fill → metrics

---

## Module Responsibilities

| Module | Responsibility | Key Functions |
|---|---|---|
| `data_samplers.py` | Sampling algorithms | `kmeans_representative_sampling`, `kcenter_greedy` |
| `ilr_functions.py` | Compositional transform | `calculate_sbp_from_hca`, `make_basis_formula` |
| `metric_functions.py` | Evaluate sample quality | `calc_ks_wasserstein`, `pca_reconstruction_error`, `plot_ecdf` |
| `geostats_cluster.py` | Spatial similarity | `compute_similarity_matrix` |

---

## R&D Norms

This is an **active research codebase**. When working here:

- **Explore alternatives before settling.** If there are multiple valid approaches, surface them with trade-offs rather than picking one silently.
- **Preserve existing implementations.** Legacy functions are kept intentionally for benchmarking and comparison — do not delete them. If adding a new version of an algorithm, add it alongside the old one.
- **Explain non-obvious math.** ILR, k-center, kernel bandwidth selection, etc. — add inline comments explaining the *why* of equations, not just the *what*.
- **Validate outputs.** New sampling functions should have a corresponding metric or sanity check (e.g., coverage distance, KS stat) to confirm behavior.
- **Flag assumptions.** If an algorithm has known edge cases or requires assumptions about the input (e.g., closed compositions, no zeros for log transforms), document them at the function level.

---

## Code Standards

- Type hints on all function signatures
- `logging` not `print()` for debug/info
- `pathlib` over `os.path`
- `black` + `ruff` formatting
- Prefer dataclasses over plain dicts for structured return types
- NumPy-first for array operations; use pandas only at the interface layer

---

## Key Design Decisions

- **ILR basis derived from HCA** (`calculate_sbp_from_hca`): Ward linkage on compositional columns to determine the sequential binary partition. This makes the basis data-driven, not arbitrary.
- **Two-stage sampling** (KMeans → KCenter): KMeans handles central tendency coverage; KCenter greedy fills spatial gaps. The combination outperforms either alone.
- **`kcenter_greedy` is the primary algorithm**: 2-approximation to the k-center problem. Prefer it over legacy methods.
- **`geostats_cluster.py` is experimental**: Not yet in the package. The similarity matrix is `O(n²)` and slow for large datasets — parallelization is an open research question.

---

## Active Development Areas

- `geostats_cluster.py` needs packaging into `src/` and performance optimization
- `ilr_functions.py` and `metric_functions.py` have uncommitted changes (check git diff before touching)
- Benchmark suite in `example.py` should be refactored into proper tests

---

## Dependencies

- `pandas >= 1.5`, `numpy >= 1.22`, `scikit-learn >= 1.2`
- `scipy`, `matplotlib`, `seaborn`
- Python >= 3.9 (prefer 3.11+)
- Package manager: **conda**
