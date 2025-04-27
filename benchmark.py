import numpy as np
from configuration import (
    NpConfiguration,
    batch_config_cost_sum_sqrt,
    batch_config_cost_linalg_norm,
    batch_config_cost,
    batch_config_cost_no_check,
    batch_config_cost_numba_sliced_dists_naive,
    batch_config_cost_numba_parallel_sliced_dists_naive,
    batch_config_cost_numba_naive_unrolled,
    batch_config_cost_numba_unrolled,
    batch_config_cost_numba_better_unrolled,
    batch_config_cost_numba_naive_unrolled_non_squared,
    batch_config_cost_numba_unrolled_non_squared,
    batch_config_cost_numba_naive_unrolled_non_squared_reduction
)
from typing import List, Callable
import pytest

batch_functions = {
    "sum_sqrt_baseline": batch_config_cost_sum_sqrt,
    "linalg_norm_baseline": batch_config_cost_linalg_norm,
    "numba_naive": batch_config_cost_numba_sliced_dists_naive,
    # "numba_parallel": batch_config_cost_numba_parallel_sliced_dists_naive,
    "numba_naive_unrolled": batch_config_cost_numba_naive_unrolled,
    "numba_naive_unrolled_non_squared": batch_config_cost_numba_naive_unrolled_non_squared,
    "numba_naive_unroll_numba_reduction": batch_config_cost_numba_naive_unrolled_non_squared_reduction
    # "in_use": batch_config_cost,
    # "baseline_no_check": batch_config_cost_no_check,
    # "numba_unrolled": batch_config_cost_numba_unrolled,
    # "numba_better_unrolled": batch_config_cost_numba_better_unrolled,
    # "numba_unrolled_non_squared": batch_config_cost_numba_unrolled_non_squared,
}

def generate_test_dat
a(dims: List, num_pts=1000):
    cumulative_dimension = sum(dims)
    slices = [(sum(dims[:i]), sum(dims[: i + 1])) for i in range(len(dims))]
    pt = np.random.rand(cumulative_dimension)
    single_config = NpConfiguration(pt, slices)

    pts = np.random.rand(num_pts, cumulative_dimension)
    return single_config, pts


@pytest.mark.parametrize("func_name, cost_fn", batch_functions.items())
@pytest.mark.parametrize("reduction", ["max", "sum"])
@pytest.mark.parametrize("dims", [[2, 2], [7, 7], [3, 3, 3], [14]])
@pytest.mark.parametrize("num_points", [1, 10, 100, 1000, 5000])
def test_batch_cost_variants(benchmark, func_name: str, cost_fn: Callable, dims, num_points, reduction):
    config, pts = generate_test_data(dims, num_pts=num_points)

    def fn(pt, pts):
        return cost_fn(pt, pts, metric="euclidean", reduction=reduction)

    benchmark.group = f"{func_name}-reduction-{reduction}-dims-{dims}-npts-{num_points}"
    benchmark(fn, config, pts)