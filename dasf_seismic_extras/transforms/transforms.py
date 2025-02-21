#!/usr/bin/env python3

import os
import shutil

import numpy as np
import pandas as pd
import segyio
import torch
import zarr
from dasf.transforms import ArrayToHDF5, ArrayToZarr, Transform
from dasf.utils.funcs import get_dask_running_client
from dasf.utils.types import is_array, is_dask_gpu_array, is_gpu_array
from segyio.tools import from_array as from_array_to_segy
from segysak import open_seisnc

from dasf_seismic_extras.utils import utils as seismic_utils

try:
    import cudf
    import cupy as cp
except ImportError:
    pass


class SEGYToArray(Transform):
    def __init__(self, save=True, filename=None, raw=False):
        self.save = save
        self.filename = filename
        self.raw = raw

    @staticmethod
    def _convert_filename(url):
        if url.endswith(".sgy"):
            return url.replace(".sgy", ".npy")
        return url + ".npy"

    def transform(self, X):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray

        from dasf_seismic_extras.datasets.base import DatasetSEGY

        if issubclass(X.__class__, DatasetSEGY):
            if self.filename:
                url = self.filename
            elif hasattr(X, '_root_file'):
                url = self._convert_filename(X._root_file)
            else:
                raise Exception("Array requires a valid path to convert "
                                    "to Array.")

            if self.save:
                np.save(url, X._data)

            # This is just a place holder
            if self.raw:
                return X._data
            return DatasetArray(download=False, name=str(X), root=url)
        else:
            raise Exception("Input is not a SEG-Y dataset.")


class ArrayToSEGY(Transform):
    def __init__(self, filename=None):
        self.filename = filename

    @staticmethod
    def _convert_filename(url):
        if url.endswith(".npy"):
            return url.replace(".npy", ".sgy")
        return url + ".sgy"

    def transform(self, X):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray

        from dasf_seismic_extras.datasets.base import DatasetSEGY

        if issubclass(X.__class__, DatasetArray) or is_array(X):
            if issubclass(X.__class__, DatasetArray):
                X = X._data

            if self.filename:
                url = self.filename
            elif hasattr(X, '_root_file'):
                url = self._convert_filename(X._root_file)
            else:
                raise Exception("Array requires a valid path to convert "
                                "to Array.")

            # XXX: Workaround to avoid error with CuPy and Zarr library
            if is_dask_gpu_array(X):
                X = X.map_blocks(lambda x: x.get())
            elif is_gpu_array(X):
                X = X.get()

            from_array_to_segy(url, X)

            # This is just a place holder
            return DatasetSEGY(download=False, name=str(X), root=url)
        else:
            raise Exception("Input is not a SEG-Y dataset.")


class SEGYToZarr(ArrayToZarr):
    def __init__(self, chunks=None, save=True, filename=None):
        super().__init__(chunks=chunks, save=save, filename=filename)

    @staticmethod
    def _convert_filename(url):
        if url.endswith(".sgy"):
            return url.replace(".sgy", ".zarr")
        return url + ".zarr"

    def _lazy_transform_generic(self, X, **kwargs):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetZarr

        from dasf_seismic_extras.datasets.base import DatasetSEGY

        name = None

        if isinstance(X, DatasetSEGY):
            name = X._name
            chunks = X.get_chunks()

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._lazy_transform_generic_all(X._data)
        else:
            raise Exception("It is not an SEG-Y type.")

        return DatasetZarr(name=name, download=False, root=url, chunks=chunks)

    def _transform_generic(self, X, **kwargs):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetZarr

        from dasf_seismic_extras.datasets.base import DatasetSEGY

        name = None
        url = None

        if hasattr(X, 'get_chunks') and \
           (X.get_chunks() is not None and X.get_chunks() != 'auto'):
            chunks = X.get_chunks()
        else:
            chunks = self.chunks

        if chunks is None:
            raise Exception("Chunks needs to be specified.")

        if isinstance(X, DatasetSEGY):
            name = X._name

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._transform_generic_all(X._data, chunks)
        else:
            raise Exception("It is not an SEG-Y type.")

        return DatasetZarr(name=name, download=False, root=url, chunks=chunks)


class ZarrToSEGY(Transform):
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def _convert_filename(url):
        if url.endswith(".zarr"):
            return url.replace(".zarr", ".sgy")
        return url + ".sgy"

    def transform(self, X):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetZarr

        from dasf_seismic_extras.datasets.base import DatasetSEGY

        if issubclass(X.__class__, DatasetZarr) or is_array(X):
            if issubclass(X.__class__, DatasetZarr):
                X = X._data

            if self.filename:
                url = self.filename
            elif hasattr(X, '_root_file'):
                url = X._root_file
            else:
                raise Exception("Array requires a valid path to convert "
                                "to Array.")

            url = self._convert_filename(url)

            from_array_to_segy(self.filename, X)

            # This is just a place holder
            return DatasetSEGY(download=False, name=str(X), root=url)
        else:
            raise Exception("Input is not a SEG-Y dataset.")


class SEGYToHDF5(ArrayToHDF5):
    def __init__(self, dataset_path, chunks=None, save=True, filename=None):
        super().__init__(dataset_path=dataset_path, chunks=chunks, save=save,
                         filename=filename)

    @staticmethod
    def _convert_filename(url):
        if url.endswith(".sgy"):
            return url.replace(".sgy", ".hdf5")
        return url + ".hdf5"

    def _lazy_transform_generic(self, X, **kwargs):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetHDF5

        from dasf_seismic_extras.datasets.base import DatasetSEGY

        name = None
        chunks = None

        if isinstance(X, DatasetSEGY):
            name = X._name
            chunks = X.get_chunks()

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._lazy_transform_generic_all(X._data)
        else:
            raise Exception("It is not an SEG-Y type.")

        return DatasetHDF5(name=name, download=False, root=url, chunks=chunks,
                           dataset_path=self.dataset_path)

    def _transform_generic(self, X, **kwargs):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetHDF5

        from dasf_seismic_extras.datasets.base import DatasetSEGY

        name = None
        url = None

        if hasattr(X, 'get_chunks') and \
           (X.get_chunks() is not None and X.get_chunks() != 'auto'):
            chunks = X.get_chunks()
        else:
            chunks = self.chunks

        if isinstance(X, DatasetSEGY):
            name = X._name

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._transform_generic_all(X._data)
        else:
            raise Exception("It is not an SEG-Y type.")

        return DatasetHDF5(name=name, download=False, root=url, chunks=chunks,
                           dataset_path=self.dataset_path)


class SEGYToDataFrame(Transform):
    def _transform_gpu(self, X):
        # XXX: Avoid circular dependency
        from dasf_seismic_extras.datasets.base import DatasetSEGY
        if isinstance(X, DatasetSEGY):
            cube = cp.array(segyio.tools.cube(X))
            datas = cp.stack(cp.ascontiguousarray(cube), axis=-1)
            return cudf.DataFrame(datas, columns=["seg-y"])
        else:
            raise Exception("It is not an SEG-Y type.")

    def _transform_cpu(self, X):
        # XXX: Avoid circular dependency
        from dasf_seismic_extras.datasets.base import DatasetSEGY
        if isinstance(X, DatasetSEGY):
            cube = np.array(segyio.tools.cube(X))
            datas = np.stack(np.ascontiguousarray(cube), axis=-1)
            return pd.DataFrame(datas, columns=["seg-y"])
        else:
            raise Exception("It is not an SEG-Y type.")


class SEGYToSeisnc(Transform):
    def transform(self, X):
        # XXX: Avoid circular dependency
        from dasf_seismic_extras.datasets.base import DatasetSEGY
        if isinstance(X, DatasetSEGY):
            seisnc_file = seismic_utils.convert_to_seisnc(X._filename)

            return open_seisnc(seisnc_file, chunks=X.chunksize)
        else:
            raise Exception("It is not an SEG-Y type.")


class SEGYToTensor(Transform):
    def transform(self, X):
        # XXX: Avoid circular dependency
        from dasf_seismic_extras.datasets.base import DatasetSEGY
        if isinstance(X, DatasetSEGY):
            return torch.Tensor(X._data)
        else:
            raise Exception("It is not an SEG-Y type.")
