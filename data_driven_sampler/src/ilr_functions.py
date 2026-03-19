import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from collections import Counter


def calculate_sbp_from_hca(df):
    """
    Derive a sequential binary partition (SBP) from hierarchical cluster analysis (HCA).

    Uses Ward linkage on the pairwise Euclidean distances of the input data to build
    a dendrogram, then walks the merge history to identify each split. Each split
    defines one ILR balance: the parts that merged from one sub-cluster form the
    numerator and the parts from the other sub-cluster form the denominator.

    Parameters
    ----------
    df : pandas.DataFrame
        Compositional data where each row is a part (component) and each column is
        a sample. The index values are used as part labels in the returned partition.

    Returns
    -------
    dict
        Mapping of split index (int) to a dict with keys:
        - ``"Numerator"`` : list of part labels in the numerator group.
        - ``"Denominator"`` : list of part labels in the denominator group.
    """
    y = pdist(df.to_numpy(), metric="euclidean")
    Z = hierarchy.ward(y)
    cut_heights = Z[:, 2][::-1]
    dct_basis = {}
    basis_choices = df.index.tolist()
    for idx, _ in enumerate(cut_heights):
        _start = fcluster(Z, cut_heights[idx], criterion="distance")
        _cnts = Counter(_start)
        if idx == len(cut_heights) - 1:
            _step = fcluster(Z, 0, criterion="distance")
        else:
            _step = fcluster(Z, cut_heights[idx + 1], criterion="distance")

        _diff = _start - _step
        for _clst, _cnt in _cnts.items():

            _idx_clsts = np.where(_start == _clst)[0]

            if _cnt < 2:
                continue
            if all(_diff[_idx_clsts] == 0):
                continue
            if len(set(_step[_idx_clsts])) == 1:
                continue

            idx_denom = set(np.where(_diff < 0)[0]).intersection(_idx_clsts)
            idx_num = set(_idx_clsts).difference(idx_denom)

            dct_basis[idx] = {
                "Numerator": [basis_choices[_bc] for _bc in idx_num],
                "Denominator": [basis_choices[_bc] for _bc in idx_denom],
            }
    return dct_basis


def product_power(df, columns):
    """
    Compute the geometric mean of selected columns for each row.

    Equivalent to the ``r``-th root of the product of the selected column values,
    where ``r`` is the number of columns. This is the geometric mean used in the
    ILR numerator/denominator sub-compositions.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing at least the columns listed in ``columns``.
    columns : list of str
        Column names whose values will be multiplied and then rooted.

    Returns
    -------
    pandas.Series
        Row-wise geometric mean of the selected columns.
    """
    n_feats = len(columns)
    _prod = df[columns].prod(axis=1)
    return _prod ** (1 / n_feats)


def constant_front(numerator, denominator):
    """
    Compute the Aitchison ILR scaling constant for a binary partition.

    Returns ``sqrt((r * s) / (r + s))``, where ``r`` is the number of parts in the
    numerator group and ``s`` is the number of parts in the denominator group.
    This constant ensures the resulting ILR coordinate is an isometric projection
    of the simplex.

    Parameters
    ----------
    numerator : list
        Parts belonging to the numerator sub-composition.
    denominator : list
        Parts belonging to the denominator sub-composition.

    Returns
    -------
    float
        The ILR scaling constant.
    """
    n_num = len(numerator)
    n_den = len(denominator)
    return np.sqrt((n_num * n_den) / (n_num + n_den))


def make_basis_formula(df, numerator, denominator):
    """
    Compute a single ILR balance coordinate for a binary partition.

    Calculates ``C * ln(geometric_mean(numerator) / geometric_mean(denominator))``,
    where ``C`` is the Aitchison scaling constant from :func:`constant_front`. This
    produces one isometric log-ratio (ILR) coordinate corresponding to the supplied
    sequential binary partition.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame whose columns are samples and whose rows contain the part values.
        Must include all columns named in ``numerator`` and ``denominator``.
    numerator : list of str
        Column names of parts that form the numerator sub-composition.
    denominator : list of str
        Column names of parts that form the denominator sub-composition.

    Returns
    -------
    pandas.Series
        ILR coordinate value for each sample (row in ``df``).
    """
    _const = constant_front(numerator, denominator)
    _num = product_power(df, numerator)
    _den = product_power(df, denominator)
    _fnc = np.log(_num / _den)
    return _const * _fnc
