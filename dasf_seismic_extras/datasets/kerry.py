#!/usr/bin/env python3

from dasf.datasets.download import DownloadWget

from dasf_seismic_extras.datasets.base import DatasetSEGY, DatasetSeismicType


class Kerry(DatasetSEGY, DownloadWget):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Kerry 3-D"
        subtype = DatasetSeismicType.prestack_seismic
        url = ("https://ml-for-seismic-data-interpretation.s3.amazonaws.com/"
               "SEG-Y/Kerry3e.sgy")
        filename = "kerry.sgy"

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadWget.__init__(self, url, filename, self._root, download)
