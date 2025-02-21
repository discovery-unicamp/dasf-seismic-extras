#!/usr/bin/env python3


from dasf.datasets.download import DownloadWget

from dasf_seismic_extras.datasets import DatasetSeismicType
from dasf_seismic_extras.datasets.base import DatasetSEGY


class Poseidon(DatasetSEGY, DownloadWget):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        url = ("https://ml-for-seismic-data-interpretation.s3.amazonaws.com/"
               "SEG-Y/Poseidon_i1000-3600_x900-3200.sgy")
        name = "Poseidon 3-D"
        filename = "poseidon.sgy"
        subtype = DatasetSeismicType.poststack_seismic

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadWget.__init__(self, url, filename, self._root, download)
