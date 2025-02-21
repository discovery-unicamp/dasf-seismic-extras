#!/usr/bin/env python3

from dasf.datasets.download import DownloadGDrive

from dasf_seismic_extras.datasets.base import DatasetSEGY, DatasetSeismicType


class Delft(DatasetSEGY, DownloadGDrive):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Delft"
        subtype = DatasetSeismicType.prestack_seismic
        google_file_id = "1Bdy9QdEVbcMr-NXkjr6881DivLUhi_kU"
        filename = "delft.sgy"

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._root, download)
