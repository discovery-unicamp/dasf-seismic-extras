#!/usr/bin/env python3

import dask
import dask.array as da
from dasf.datasets import DatasetArray
from dasf.datasets.download import DownloadGDrive, DownloadWget

from dasf_seismic_extras.datasets import DatasetSeismicType
from dasf_seismic_extras.datasets.base import DatasetSEGY


class ParihakaSkel(DatasetSEGY, DownloadWget):
    def __init__(self,
                 name,
                 subtype,
                 url=None,
                 filename=None,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadWget.__init__(self, url, filename,
                              self._root, download)


class ParihakaFull(ParihakaSkel):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Parihaka Full 3-D"
        subtype = DatasetSeismicType.full_stack
        url = ("https://ml-for-seismic-data-interpretation.s3.amazonaws.com/"
               "SEG-Y/Parihaka_PSTM_full-3D.sgy")
        filename = "parihaka_full.sgy"

        super().__init__(name, subtype=subtype, url=url, filename=filename,
                         download=download, root=root, chunks=chunks,
                         contiguous=contiguous)


class ParihakaNear(ParihakaSkel):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Parihaka Near 3-D"
        subtype = DatasetSeismicType.near_stack
        url = ("http://s3.amazonaws.com/open.source.geoscience/open_data/"
               "newzealand/Taranaiki_Basin/PARIHAKA-3D/"
               "Parihaka_PSTM_near_stack.sgy")
        filename = "parihaka_near.sgy"

        super().__init__(name, subtype=subtype, url=url, filename=filename,
                         download=download, root=root, chunks=chunks,
                         contiguous=contiguous)


class ParihakaMid(ParihakaSkel):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Parihaka Mid 3-D"
        subtype = DatasetSeismicType.mid_stack
        url = ("http://s3.amazonaws.com/open.source.geoscience/open_data/"
               "newzealand/Taranaiki_Basin/PARIHAKA-3D/"
               "Parihaka_PSTM_mid_stack.sgy")
        filename = "parihaka_mid.sgy"

        super().__init__(name, subtype=subtype, url=url, filename=filename,
                         download=download, root=root, chunks=chunks,
                         contiguous=contiguous)


class ParihakaFar(ParihakaSkel):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Parihaka Far 3-D"
        subtype = DatasetSeismicType.far_stack
        url = ("http://s3.amazonaws.com/open.source.geoscience/open_data/"
               "newzealand/Taranaiki_Basin/PARIHAKA-3D/"
               "Parihaka_PSTM_far_stack.sgy")
        filename = "parihaka_far.sgy"

        super().__init__(name, subtype=subtype, url=url, filename=filename,
                         download=download, root=root, chunks=chunks,
                         contiguous=contiguous)


class ParihakaTrain(DatasetArray, DownloadGDrive):
    def __init__(self, download=False, root=None, chunks="auto"):
        name = "Parihaka Full 3-D Train Data"
        filename = "parihaka_full_data_train.npz"
        subtype = DatasetSeismicType.full_stack
        google_file_id = "1sJwFkjnIde11Ak_E2wY4N3MZ31BFMbDA"

        DatasetArray.__init__(self, name, subtype=subtype, download=download,
                              root=root, chunks=chunks)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._cache_dir)

    def _lazy_load(self, xp):
        npy_shape = self.shape

        self._data = dask.delayed(xp.load)(self._root, allow_pickle=True,
                                           mmap_mode='r')

        # Parihaka Train test has only `data` key
        self._data = da.from_delayed(self._data["data"], shape=npy_shape,
                                     dtype=xp.float32)

        if isinstance(self.__chunks, tuple):
            self._data = self._data.rechunk(self.__chunks)

    def _load(self, xp):
        self._data = xp.load(self._root, allow_pickle=True,
                             mmap_mode='r')

        # Parihaka Train test has only `data` key
        self._data = self._data["data"]


class ParihakaTrainLabels(DatasetArray, DownloadGDrive):
    def __init__(self, download=False, root=None, chunks="auto"):
        name = "Parihaka Full 3-D Train Labels"
        filename = "parihaka_full_labels_train.npz"
        subtype = DatasetSeismicType.full_stack
        google_file_id = "1nTOw0S8otG60ZslWTPqOzB3dkB-ZyJkj"

        DatasetArray.__init__(self, name, subtype=subtype, download=download,
                              root=root, chunks=chunks)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._cache_dir)

    def _lazy_load(self, xp):
        npy_shape = self.shape

        self._data = dask.delayed(xp.load)(self._root, allow_pickle=True,
                                           mmap_mode='r')

        # Parihaka Train test has only `labels` key
        self._data = da.from_delayed(self._data["labels"], shape=npy_shape,
                                     dtype=xp.float32)

        if isinstance(self.__chunks, tuple):
            self._data = self._data.rechunk(self.__chunks)

    def _load(self, xp):
        self._data = xp.load(self._root, allow_pickle=True, mmap_mode='r')
        # Parihaka Train test has only `labels` key
        self._data = self._data["labels"]


class ParihakaDataTest1(DatasetArray, DownloadGDrive):
    def __init__(self, download=False, root=None, chunks="auto"):
        name = "Parihaka Full 3-D Data Test 1"
        filename = "parihaka_full_data_test_1.npz"
        subtype = DatasetSeismicType.full_stack
        google_file_id = "1ZwVLKYExikPh0Cu-xNNmbbH-4vMrRZBL"

        DatasetArray.__init__(self, name, subtype=subtype, download=download,
                              root=root, chunks=chunks)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._cache_dir)

    def _lazy_load(self, xp):
        npy_shape = self.shape

        self._data = dask.delayed(xp.load)(self._root, allow_pickle=True,
                                           mmap_mode='r')

        # Parihaka Train test has only `data` key
        self._data = da.from_delayed(self._data["data"], shape=npy_shape,
                                     dtype=xp.float32)

        if isinstance(self.__chunks, tuple):
            self._data = self._data.rechunk(self.__chunks)

    def _load(self, xp):
        self._data = xp.load(self._root, allow_pickle=True, mmap_mode='r')
        # Parihaka Train test has only `data` key
        self._data = self._data["data"]


class ParihakaDataTest2(DatasetArray, DownloadGDrive):
    def __init__(self, download=False, root=None, chunks="auto"):
        name = "Parihaka Full 3-D Data Test 2"
        filename = "parihaka_full_data_test_2.npz"
        subtype = DatasetSeismicType.full_stack
        google_file_id = "1vTRGmAsUMi2rdW7AofwP1EZVVTYzjBWY"

        DatasetArray.__init__(self, name, subtype=subtype, download=download,
                              root=root, chunks=chunks)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._cache_dir)

    def _lazy_load(self, xp):
        npy_shape = self.shape

        self._data = dask.delayed(xp.load)(self._root, allow_pickle=True,
                                           mmap_mode='r')

        # Parihaka Train test has only `data` key
        self._data = da.from_delayed(self._data["data"], shape=npy_shape,
                                     dtype=xp.float32)

        if isinstance(self.__chunks, tuple):
            self._data = self._data.rechunk(self.__chunks)

    def _load(self, xp):
        self._data = xp.load(self._root, allow_pickle=True, mmap_mode='r')
        # Parihaka Train test has only `data` key
        self._data = self._data["data"]
