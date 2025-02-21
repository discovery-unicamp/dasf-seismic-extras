#!/usr/bin/env python3

from dasf.datasets.download import DownloadWget

from dasf_seismic_extras.datasets import DatasetSeismicType
from dasf_seismic_extras.datasets.base import DatasetSEGY


class Stratton3D(DatasetSEGY, DownloadWget):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        url = ("https://s3.amazonaws.com/open.source.geoscience/open_data/"
               "stratton/segy/processed/Stratton3d.sgy")
        name = "Stratton 3-D"
        filename = "stratton_3d.sgy"
        subtype = DatasetSeismicType.prestack_seismic

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadWget.__init__(self, url, filename, self._root, download)
