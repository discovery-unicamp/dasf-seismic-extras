#!/usr/bin/env python3

from dasf.datasets.base import DatasetDataFrame
from dasf.datasets.download import DownloadGDrive

from dasf_seismic_extras.datasets import DatasetSeismicType


class PanomaCouncilGroveFieldFaciesWellLogs(DatasetDataFrame, DownloadGDrive):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto"):

        name = "Panoma Council Grove Field Facies Well Logs"
        filename = "facies_data.csv"
        google_file_id = "1A-Xge2FkEzVY8-vGVZJNUcOOdHEKtcJ3"
        self._subtype = DatasetSeismicType.none

        DatasetDataFrame.__init__(self, name=name, download=False,
                                  root=root, chunks=chunks)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._root, download)
