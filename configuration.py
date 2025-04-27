import numpy as np
import numba

from typing import List, Tuple, Union
from numpy.typing import NDArray

class NpConfiguration:
    __slots__ = (
        "array_slice",
        "q",
        "_num_agents",
    )  # , "robot_views"

    array_slice: NDArray
    # slice: List[Tuple[int, int]]
    q: NDArray
    _num_agents: int

    def __init__(self, q: NDArray, _slice: List[Tuple[int, int]]):
        self.array_slice = np.array(_slice)
        self.q = q.astype(np.float64)

        self._num_agents = len(self.array_slice)

    def num_agents(self) -> int:
        return self._num_agents

    def __setitem__(self, ind, data):
        s, e = self.array_slice[ind]
        self.q[s:e] = data

    def state(self) -> NDArray:
        return self.q


@numba.jit((numba.float64[:, :], numba.int64[:, :]), nopython=True, parallel=True)
def numba_parallelized_sum(
    squared_diff: NDArray, slices: NDArray
) -> NDArray:
    num_agents = len(slices)
    dists = np.zeros((num_agents, squared_diff.shape[0]))

    for i in numba.prange(num_agents):
        s, e = slices[i]
        dists[i, :] = np.sqrt(np.sum(squared_diff[:, s:e], axis=1))

    return dists


@numba.jit((numba.float64[:, :], numba.int64[:, :]), nopython=True, fastmath=True)
def compute_sliced_dists_naive(squared_diff: NDArray, slices: NDArray) -> NDArray:
    num_slices = len(slices)
    num_samples = squared_diff.shape[0]
    dists = np.empty((num_slices, num_samples), dtype=np.float64)

    for i in range(num_slices):
        s, e = slices[i]
        dists[i, :] = np.sqrt(np.sum(squared_diff[:, s:e], axis=1))

    return dists

@numba.jit((numba.float64[:, :], numba.int64[:, :]), nopython=True, fastmath=True, parallel=True)
def compute_sliced_dists_naive_unrolled(squared_diff: NDArray, slices: NDArray) -> NDArray:
    """Compute Euclidean distances for sliced configurations with optimizations."""
    num_slices = len(slices)
    num_samples = squared_diff.shape[0]
    dists = np.empty((num_slices, num_samples), dtype=np.float64)

    # Process each slice independently
    for i in range(num_slices):
        s, e = slices[i]
        slice_width = e - s

        # Optimize the inner loop for better vectorization and cache usage
        for j in range(num_samples):
            sum_squared = 0.0
            # For larger slices, use a regular loop which Numba can vectorize
            for k in range(s, e):
                sum_squared += squared_diff[j, k]

            dists[i, j] = np.sqrt(sum_squared)

    return dists

@numba.jit((numba.float64[:, :], numba.int64[:, :]), nopython=True, fastmath=True, parallel=True)
def compute_sliced_dists_naive_unrolled_non_squared(diff: NDArray, slices: NDArray) -> NDArray:
    """Compute Euclidean distances for sliced configurations with optimizations."""
    num_slices = len(slices)
    num_samples = diff.shape[0]
    dists = np.empty((num_slices, num_samples), dtype=np.float64)

    # Process each slice independently
    for i in range(num_slices):
        s, e = slices[i]
        slice_width = e - s

        # Optimize the inner loop for better vectorization and cache usage
        for j in range(num_samples):
            sum_squared = 0.0
            # For larger slices, use a regular loop which Numba can vectorize
            for k in range(s, e):
                sum_squared += diff[j, k] * diff[j, k]

            dists[i, j] = np.sqrt(sum_squared)

    return dists

@numba.jit((numba.float64[:, :], numba.int64[:, :]), nopython=True, fastmath=True, parallel=True)
def compute_sliced_dists_unrolled(squared_diff: NDArray, slices: NDArray) -> NDArray:
    """Compute Euclidean distances for sliced configurations with optimizations."""
    num_slices = len(slices)
    num_samples = squared_diff.shape[0]
    dists = np.empty((num_slices, num_samples), dtype=np.float64)

    # Process each slice independently
    for i in range(num_slices):
        s, e = slices[i]
        slice_width = e - s

        # Optimize the inner loop for better vectorization and cache usage
        for j in range(num_samples):
            sum_squared = 0.0
            # Unroll the loop for small slices or use direct sum for better performance
            if slice_width <= 8:  # Threshold for loop unrolling
                # Manual loop unrolling for small dimensions
                idx = s
                while idx + 4 <= e:  # Process 4 elements at a time
                    sum_squared += squared_diff[j, idx] + squared_diff[j, idx+1] + squared_diff[j, idx+2] + squared_diff[j, idx+3]
                    idx += 4
                # Handle remaining elements
                while idx < e:
                    sum_squared += squared_diff[j, idx]
                    idx += 1
            else:
                # For larger slices, use a regular loop which Numba can vectorize
                for k in range(s, e):
                    sum_squared += squared_diff[j, k]

            dists[i, j] = np.sqrt(sum_squared)

    return dists


@numba.jit((numba.float64[:, :], numba.int64[:, :]), nopython=True, fastmath=True)
def compute_sliced_dists_better_unrolled(squared_diff: NDArray, slices: NDArray) -> NDArray:
    """Compute Euclidean distances for sliced configurations - optimized for sequential execution."""
    num_slices = len(slices)
    num_samples = squared_diff.shape[0]
    dists = np.empty((num_slices, num_samples), dtype=np.float64)

    # Pre-extract slice bounds for better access patterns
    starts = np.empty(num_slices, dtype=np.int64)
    ends = np.empty(num_slices, dtype=np.int64)
    for i in range(num_slices):
        starts[i] = slices[i, 0]
        ends[i] = slices[i, 1]

    # Process samples in the outer loop for better cache locality
    for j in range(num_samples):
        # Process each slice for this sample
        for i in range(num_slices):
            s = starts[i]
            e = ends[i]
            sum_squared = 0.0

            # Improved loop unrolling with early exit conditions
            # Unroll by 8 when possible
            idx = s
            while idx + 8 <= e:
                sum_squared += (
                    squared_diff[j, idx]
                    + squared_diff[j, idx + 1]
                    + squared_diff[j, idx + 2]
                    + squared_diff[j, idx + 3]
                    + squared_diff[j, idx + 4]
                    + squared_diff[j, idx + 5]
                    + squared_diff[j, idx + 6]
                    + squared_diff[j, idx + 7]
                )
                idx += 8

            # Unroll remaining by 4
            while idx + 4 <= e:
                sum_squared += (
                    squared_diff[j, idx]
                    + squared_diff[j, idx + 1]
                    + squared_diff[j, idx + 2]
                    + squared_diff[j, idx + 3]
                )
                idx += 4

            # Handle remaining elements
            while idx < e:
                sum_squared += squared_diff[j, idx]
                idx += 1

            dists[i, j] = np.sqrt(sum_squared)

    return dists


@numba.jit((numba.float64[:, :], numba.int64[:, :]), nopython=True, fastmath=True)
def compute_sliced_euclidean_dists(diff: NDArray, slices: NDArray) -> NDArray:
    """Compute Euclidean distances for sliced configurations - optimized for sequential execution."""
    num_slices = len(slices)
    num_samples = diff.shape[0]
    dists = np.empty((num_slices, num_samples), dtype=np.float64)

    # Pre-extract slice bounds for better access patterns
    starts = np.empty(num_slices, dtype=np.int64)
    ends = np.empty(num_slices, dtype=np.int64)
    for i in range(num_slices):
        starts[i] = slices[i, 0]
        ends[i] = slices[i, 1]

    # Process samples in the outer loop for better cache locality
    for j in range(num_samples):
        # Process each slice for this sample
        for i in range(num_slices):
            s = starts[i]
            e = ends[i]
            sum_squared = 0.0

            # Improved loop unrolling with early exit conditions
            # Unroll by 8 when possible
            idx = s
            while idx + 8 <= e:
                sum_squared += (
                    diff[j, idx] * diff[j, idx]
                    + diff[j, idx + 1] * diff[j, idx + 1]
                    + diff[j, idx + 2] * diff[j, idx + 2]
                    + diff[j, idx + 3] * diff[j, idx + 3]
                    + diff[j, idx + 4] * diff[j, idx + 4]
                    + diff[j, idx + 5] * diff[j, idx + 5]
                    + diff[j, idx + 6] * diff[j, idx + 6]
                    + diff[j, idx + 7] * diff[j, idx + 7]
                )
                idx += 8

            # Unroll remaining by 4
            while idx + 4 <= e:
                sum_squared += (
                    diff[j, idx] * diff[j, idx]
                    + diff[j, idx + 1] * diff[j, idx + 1]
                    + diff[j, idx + 2] * diff[j, idx + 2]
                    + diff[j, idx + 3] * diff[j, idx + 3]
                )
                idx += 4

            # Handle remaining elements
            while idx < e:
                sum_squared += diff[j, idx] * diff[j, idx]
                idx += 1

            dists[i, j] = np.sqrt(sum_squared)

    return dists


# @numba.jit(numba.float64[:](numba.float64[:, :]), nopython=True, fastmath=True)
# def compute_sum_reduction(dists: NDArray) -> NDArray:
#     """Compute sum reduction across robot distances."""
#     return np.sum(dists, axis=0)


@numba.jit(numba.float64[:](numba.float64[:, :]), nopython=True, fastmath=True)
def compute_sum_reduction(dists: NDArray) -> NDArray:
    """Compute sum reduction across robot distances."""
    num_slices, num_samples = dists.shape
    result = np.empty(num_samples, dtype=np.float64)

    # Manually compute sum along axis 0
    for j in range(num_samples):
        sum_val = 0.0
        for i in range(num_slices):
            sum_val += dists[i, j]
        result[j] = sum_val

    return result


@numba.jit(
    numba.float64[:](numba.float64[:, :], numba.float64), nopython=True, fastmath=True
)
def compute_max_sum_reduction(dists: NDArray, w: float) -> NDArray:
    """Compute max + w*sum reduction across robot distances."""
    num_slices, num_samples = dists.shape
    result = np.empty(num_samples, dtype=np.float64)

    # Manually compute max along axis 0
    for j in range(num_samples):
        max_val = dists[0, j]
        sum_val = dists[0, j]
        for i in range(1, num_slices):
            if dists[i, j] > max_val:
                max_val = dists[i, j]
            sum_val += dists[i, j]
        result[j] = max_val + w * sum_val

    return result


@numba.jit(numba.float64[:](numba.float64[:, :]), nopython=True, fastmath=True)
def compute_abs_max_reduction(dists: NDArray) -> NDArray:
    """Compute the maximum absolute value along axis 1 for each row."""
    num_rows, num_cols = dists.shape
    result = np.empty(num_rows, dtype=np.float64)

    for i in range(num_rows):
        # Start with first element
        max_val = abs(dists[i, 0])

        # Find maximum absolute value in the row
        for j in range(1, num_cols):
            abs_val = abs(dists[i, j])
            if abs_val > max_val:
                max_val = abs_val

        result[i] = max_val

    return result


# @numba.jit(numba.float64[:](numba.float64[:, :]), nopython=True, fastmath=True)
# def compute_max_reduction(dists: NDArray) -> NDArray:
#     """Compute max + w*sum reduction across robot distances."""
#     result = np.max(dists, axis=0)

#     return result


@numba.jit(numba.float64[:](numba.float64[:, :]), nopython=True, fastmath=True)
def compute_max_reduction(dists: NDArray) -> NDArray:
    """Compute max + w*sum reduction across robot distances."""
    num_slices, num_samples = dists.shape
    result = np.empty(num_samples, dtype=np.float64)

    # Manually compute max along axis 0
    for j in range(num_samples):
        max_val = dists[0, j]
        for i in range(1, num_slices):
            if dists[i, j] > max_val:
                max_val = dists[i, j]
        result[j] = max_val

    return result

def batch_config_cost_linalg_norm(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        all_robot_dists = np.zeros((starts._num_agents, diff.shape[0]))
        for i, (s, e) in enumerate(starts.array_slice):
            # Use sqrt(sum(x^2)) instead of np.linalg.norm
            # and pre-computed squared differences
            all_robot_dists[i, :] = np.linalg.norm(diff[:, s:e], axis=1)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + w * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)

def batch_config_cost_sum_sqrt(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        squared_diff = diff * diff

        all_robot_dists = np.zeros((starts._num_agents, diff.shape[0]))
        for i, (s, e) in enumerate(starts.array_slice):
            # Use sqrt(sum(x^2)) instead of np.linalg.norm
            # and pre-computed squared differences
            all_robot_dists[i, :] = np.sqrt(np.sum(squared_diff[:, s:e], axis=1))
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + w * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)


def batch_config_cost_numba_sliced_dists_naive(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        squared_diff = diff * diff
        all_robot_dists = compute_sliced_dists_naive(squared_diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + w * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)

def batch_config_cost_numba_parallel_sliced_dists_naive(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        squared_diff = diff * diff
        all_robot_dists = numba_parallelized_sum(squared_diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + w * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)

def batch_config_cost_numba_naive_unrolled(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        squared_diff = diff * diff
        all_robot_dists = compute_sliced_dists_naive_unrolled(squared_diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + w * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)

def batch_config_cost_numba_naive_unrolled_non_squared(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        all_robot_dists = compute_sliced_dists_naive_unrolled_non_squared(diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + w * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)

def batch_config_cost_numba_naive_unrolled_non_squared_reduction(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        all_robot_dists = compute_sliced_dists_naive_unrolled_non_squared(diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return compute_max_sum_reduction(all_robot_dists, w)
    elif reduction == "sum":
        return compute_sum_reduction(all_robot_dists)

def batch_config_cost_numba_unrolled(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        squared_diff = diff * diff
        all_robot_dists = compute_sliced_dists_unrolled(squared_diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + w * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)

def batch_config_cost_numba_better_unrolled(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        squared_diff = diff * diff
        all_robot_dists = compute_sliced_dists_better_unrolled(squared_diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + w * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)

def batch_config_cost_numba_unrolled_non_squared(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        all_robot_dists = compute_sliced_euclidean_dists(diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return np.max(all_robot_dists, axis=0) + w * np.sum(all_robot_dists, axis=0)
    elif reduction == "sum":
        return np.sum(all_robot_dists, axis=0)


def batch_config_cost(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """

    if isinstance(starts, NpConfiguration) and isinstance(batch_other, np.ndarray):
        diff = starts.state() - batch_other
        # all_robot_dists = np.zeros((starts._num_agents, diff.shape[0]))
        agent_slices = starts.array_slice
    else:
        diff = np.array([start.q.state() for start in starts]) - np.array(
            [other.q.state() for other in batch_other], dtype=np.float64
        )
        # all_robot_dists = np.zeros((starts[0].q._num_agents, diff.shape[0]))
        agent_slices = starts[0].q.array_slice

    if metric == "euclidean":
        # squared_diff = diff * diff
        all_robot_dists = compute_sliced_euclidean_dists(diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return compute_max_sum_reduction(all_robot_dists, w)
    elif reduction == "sum":
        return compute_sum_reduction(all_robot_dists)

def batch_config_cost_no_check(
    starts: Union[NpConfiguration, List[NpConfiguration]],
    batch_other: Union[NDArray, List[NpConfiguration]],
    metric: str = "max",
    reduction: str = "max",
    w: float = 0.01,
) -> NDArray:
    """Computes the cost between two lists of configurations.
    - Possible values for the metric are ['max', 'euclidean']
    - Possible values for the reduction are ['max', 'sum']
    """
    diff = starts.state() - batch_other
    agent_slices = starts.array_slice

    if metric == "euclidean":
        # squared_diff = diff * diff
        all_robot_dists = compute_sliced_euclidean_dists(diff, agent_slices)
    else:
        all_robot_dists = np.zeros((len(agent_slices), diff.shape[0]))
        for i, (s, e) in enumerate(agent_slices):
            all_robot_dists[i, :] = np.max(np.abs(diff[:, s:e]), axis=1)

    if reduction == "max":
        return compute_max_sum_reduction(all_robot_dists, w)
    elif reduction == "sum":
        return compute_sum_reduction(all_robot_dists)