#!/usr/bin/env python3

import dask.array as da
import numpy as np
from scipy import ndimage as ndi

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cundi
except ImportError:
    pass

from dasf.transforms import Transform
from dasf_seismic.utils.utils import dask_overlap, dask_trim_internal


class Gaussian(Transform):
    def __init__(self, sigmas=(1, 1, 1), truncate=4.0):
        super().__init__()

        self._sigmas = sigmas
        self._truncate = truncate

    def _lazy_transform(self, X, xndi):
        # Generate Dask Array as necessary and perform algorithm
        radius = np.round(np.max(self._sigmas) * self._truncate)
        kernel = tuple(np.full(len(self._sigmas), 2*radius + 1, 'int'))

        X_da = dask_overlap(X, kernel)

        result = X_da.map_blocks(xndi.gaussian_filter, sigma=self._sigmas,
                                 truncate=self._truncate, dtype=X.dtype)

        result = dask_trim_internal(result, kernel)
        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, xndi, xp):
        result = xndi.gaussian_filter(X, sigma=self._sigmas, truncate=self._truncate)

        result[xp.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cundi)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, ndi)

    def _transform_gpu(self, X):
        return self._transform(X, cundi, cp)

    def _transform_cpu(self, X):
        return self._transform(X, ndi, np)


class Median(Transform):
    def __init__(self, kernel=(3, 3, 3)):
        super().__init__()

        self._kernel = kernel

    def _lazy_transform(self, X, xndi):
        # Generate Dask Array as necessary and perform algorithm
        X_da = dask_overlap(X, self._kernel)

        result = X_da.map_blocks(xndi.median_filter, size=self._kernel,
                                 dtype=X.dtype)

        result = dask_trim_internal(result, self._kernel)
        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, xndi, xp):
        result = xndi.median_filter(X, size=self._kernel)

        result[xp.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cundi)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, ndi)

    def _transform_gpu(self, X):
        return self._transform(X, cundi, cp)

    def _transform_cpu(self, X):
        return self._transform(X, ndi, np)


class Convolution(Transform):
    def __init__(self, kernel=(3, 3, 3)):
        super().__init__()

        self._kernel = kernel

    def _lazy_transform(self, X, xndi):
        # Generate Dask Array as necessary and perform algorithm
        X_da = dask_overlap(X, self._kernel)

        result = X_da.map_blocks(xndi.uniform_filter, size=self._kernel,
                                 dtype=X.dtype)

        result = dask_trim_internal(result, self._kernel)
        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, xndi, xp):
        result = xndi.uniform_filter(X, size=self._kernel)

        result[xp.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cundi)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, ndi)

    def _transform_gpu(self, X):
        return self._transform(X, cundi, cp)

    def _transform_cpu(self, X):
        return self._transform(X, ndi, np)
