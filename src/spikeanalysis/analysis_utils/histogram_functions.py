import numpy as np
from numba import jit
import numba


@jit(nopython=True, cache=True)
def spike_times_to_bins(
    time_stamps: np.array, events: np.array, bin_size: np.int64, start: np.int64, end: np.int64
) -> tuple[np.array, np.array]:
    step_number = int(abs((end - start) / bin_size) + 1)
    bin_borders = np.linspace(start, end, step_number)
    bin_number = len(bin_borders) - 1
    bin_array = np.zeros((len(events), bin_number), np.int32)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    if len(time_stamps) == 0:
        return bin_array, bin_centers

    for n in range(len(events)):
        bin_array[n], _ = np.histogram(time_stamps, bin_borders + events[n])
    return bin_array, bin_centers


def rasterize(time_stamps: np.array) -> tuple[np.array, np.array]:
    x_out = np.empty((len(time_stamps) * 3))
    x_out[:] = np.NaN

    x_out[0:-1:3] = time_stamps
    x_out[1:-1:3] = time_stamps
    y_out = np.empty((len(time_stamps) * 3))
    y_out[:] = np.NaN
    y_out[0:-1:3] = 0
    y_out[1:-1:3] = 1
    xx = x_out.reshape(1, len(x_out))
    yy = y_out.reshape(1, len(y_out))

    return xx, yy


@jit(nopython=True)
def check_order(data1: np.array, ndata1: int, data2: np.array, ndata2: int) -> int:
    for index_i in range(1, ndata1):
        if data1[index_i] < data1[index_i - 1]:
            return 0
    for index_j in range(1, ndata2):
        if data2[index_j] < data2[index_j - 1]:
            return 0
    return 1


@jit(nopython=True)
def findext(data1: np.array, ndata1: int, data2: np.array, ndata2: int) -> tuple[float, float]:
    min_value = data1[0] - data2[0]
    max_value = data1[0] - data2[0]
    for ind_i in range(1, ndata1):
        for ind_j in range(1, ndata2):
            diff = data1[ind_i] - data2[ind_j]
            if diff < min_value:
                min_value = diff
            if diff > max_value:
                max_value = diff
    return min_value, max_value


@jit(nopython=True)
def reghist(
    data1: np.array,
    ndata1: int,
    data2: np.array,
    ndata2: int,
    min_value: float,
    size: float,
    nbins: float,
    counts: np.array,
) -> np.array:
    max_value = min_value + size * nbins

    for ind_i in range(ndata1):
        for ind_j in range(ndata2):
            diff = data1[ind_i] - data2[ind_j]
            if diff < min_value or diff >= max_value:
                continue
            idx = (diff - min_value) / size
            counts[int(idx)] += 1

    return counts


@jit(nopython=True)
def ordhist(
    data1: np.array,
    ndata1: int,
    data2: np.array,
    ndata2: int,
    min_value: float,
    size: float,
    nbins: int,
    counts: np.array,
) -> np.array:
    max_value = min_value + size * nbins
    j_min = 0
    for ind_i in range(ndata1):
        for ind_j in range(j_min, ndata2):
            if (data1[ind_i] - data2[ind_j]) <= max_value:
                j_min = ind_j
                break
        for ind_j in range(j_min, ndata2):
            diff = data1[ind_i] - data2[ind_j]
            if diff >= min_value:
                counts[int((diff - min_value) / size)] += 1
                if diff == len(counts):
                    counts[len(counts) - 1] += 1
    return counts


@jit(nopython=True)
def binhist(
    data1: np.array,
    ndata1: int,
    data2: np.array,
    ndata2: int,
    bins: np.array,
    nbins: int,
    counts: np.array,
) -> np.array:
    for ind_i in range(ndata1):
        for ind_j in range(ndata2):
            for ind_k in range(nbins):
                if data1[ind_i] - data2[ind_j] >= bins[ind_k] and (data1[ind_i] - data2[ind_j]) > bins[ind_k + 1]:
                    counts[ind_k] += 1
    return counts


@jit(nopython=True)
def histdiff(time_stamps: np.array, events: np.array, bin_borders: np.array) -> tuple[np.array, np.array]:
    data1 = time_stamps
    ndata1 = len(data1)
    data2 = events
    ndata2 = len(data2)
    nbins = len(bin_borders)
    print(nbins)

    if nbins == 1:
        min_value, max_value = findext(data1, ndata1, data2, ndata2)
        size = (max_value - min_value) / nbins
    else:
        nbins = len(bin_borders) - 1
        bins = bin_borders
        size = bins[1] - bins[0]

        for index in range(1, nbins):
            if abs(bins[index + 1] - bins[index] - size) > (1e-3 * size):
                size = 0
                break
        if size:
            min_value = bins[0]

    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    counts = np.zeros((nbins), dtype=np.int32)

    if size:
        if check_order(data1, ndata1, data2, ndata2):
            counts = ordhist(data1, ndata1, data2, ndata2, min_value, size, nbins, counts)
        else:
            counts = reghist(data1, ndata1, data2, ndata2, min_value, size, nbins, counts)
    else:
        counts = binhist(data1, ndata1, data2, ndata2, bins, nbins, counts)

    return counts, bin_centers


@jit(nopython=True)
def convert_to_new_bins(
    array: numba.int32[:, :, :],
    bin_number: np.int32,
) -> np.array:
    new_array = np.zeros((np.shape(array)[0], np.shape(array)[1], bin_number), dtype=np.int32)
    bin_modulo = int(np.shape(array)[2] / bin_number)

    for idx in range(np.shape(array)[0]):
        for idy in range(np.shape(array)[1]):
            idk = 0
            ikj = 0
            for idz in range(np.shape(array)[2]):
                if idz % bin_modulo == 0 and idz != 0:
                    new_array[idx, idy, ikj] = np.sum(array[idx, idy, (idk):(idz)])
                    idk = idz
                    ikj += 1
                elif idz == np.shape(array)[2] - 1:
                    new_array[idx, idy, -1] = np.sum(array[idx, idy, idk:])

    return new_array


@jit(nopython=True)
def convert_bins(bins: np.array, bin_number: np.int32) -> np.array:
    new_bins = np.zeros((bin_number,))

    old_length = np.shape(bins)[0]
    bin_keeps = np.int32(old_length / bin_number)
    if bin_keeps <= 1:
        raise Exception("fail")
    idy = 0
    for idx in range(len(bins)):
        if idx % bin_keeps == 0 and idx != 0:
            new_bins[idy] = round(bins[idx], 5)
            idy += 1

    return new_bins


@jit(nopython=True)
def z_score_values(z_trial: numba.float32[:, :, :], mean_fr: numba.float32[:], std_fr: numba.float32[:]) -> np.array:
    z_trials = np.zeros(np.shape(z_trial))
    for idx in range(len(mean_fr)):
        for idy in range(np.shape(z_trial)[1]):
            z_trials[idx, idy, :] = (z_trial[idx, idy] - mean_fr[idx]) / std_fr[idx]

    return z_trials
