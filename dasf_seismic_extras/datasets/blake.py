#!/usr/bin/env python3

from dasf.datasets.download import DownloadGDrive

from dasf_seismic_extras.datasets.base import DatasetSEGY, DatasetSeismicType


class BlakeRidgeHydrates(DatasetSEGY, DownloadGDrive):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Blake Ridge Hydrates"
        subtype = DatasetSeismicType.prestack_seismic
        google_file_id = "1bykzN8ED5W3MCtw1EXMKXvlZGslGxOG8"
        filename = "blake.sgy"

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._root, download)
