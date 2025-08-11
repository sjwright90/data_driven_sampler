import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from collections import Counter


def calculate_sbp_from_hca(df):
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
    n_feats = len(columns)
    _prod = df[columns].prod(axis=1)
    return _prod ** (1 / n_feats)


def constant_front(numerator, denominator):
    n_num = len(numerator)
    n_den = len(denominator)
    return np.sqrt((n_num * n_den) / (n_num + n_den))


def make_basis_formula(df, numerator, denominator):
    """
    Create a formula string from the numerator and denominator lists.
    """
    _const = constant_front(numerator, denominator)
    _num = product_power(df, numerator)
    _den = product_power(df, denominator)
    _fnc = np.log(_num / _den)
    return _const * _fnc
