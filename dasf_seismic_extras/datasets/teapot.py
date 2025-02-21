#!/usr/bin/env python3

from dasf.datasets.download import DownloadWget

from dasf_seismic_extras.datasets import DatasetSeismicType
from dasf_seismic_extras.datasets.base import DatasetSEGY


class Teapot3D(DatasetSEGY, DownloadWget):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        url = "http://s3.amazonaws.com/teapot/filt_mig.sgy"
        name = "Teapot 3-D"
        filename = "teapot_3d.sgy"
        subtype = DatasetSeismicType.migrated_volume

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadWget.__init__(self, url, filename, self._root, download)


class Teapot3DCMP(DatasetSEGY, DownloadWget):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        url = "http://s3.amazonaws.com/teapot/npr3_gathers.sgy"
        name = "Teapot 3-D CMPs"
        filename = "teapot_3d_cmp.sgy"
        subtype = DatasetSeismicType.cmp_gathers

        DatasetSEGY.__init__(self, name, subtype=subtype, download=download,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadWget.__init__(self, url, filename, self._cache_dir)


class Teapot3DRawCMP(DatasetSEGY, DownloadWget):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        url = "http://s3.amazonaws.com/teapot/npr3_field.sgy"
        name = "Teapot 3-D Raw CMPs"
        filename = "teapot_3d.sgy"
        subtype = DatasetSeismicType.cmp_gathers

        DatasetSEGY.__init__(self, name, subtype=subtype, download=download,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadWget.__init__(self, url, filename, self._cache_dir)
