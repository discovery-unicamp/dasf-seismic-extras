#!/usr/bin/env python3

import os
from os.path import join as pjoin

from dasf.datasets.download import DownloadGDrive

from dasf_seismic_extras.datasets import DatasetSeismicType
from dasf_seismic_extras.datasets.base import DatasetSEGY


class F3(DatasetSEGY, DownloadGDrive):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "F3 Netherlands"
        subtype = DatasetSeismicType.migrated_volume
        google_file_id = "1ZPsRasiCs1NfN72_Skn9YUgfipZ5KD4D"
        filename = "f3.sgy"

        # SEGY don't need to be downloaded, it should be after.
        DatasetSEGY.__init__(self, name=name, subtype=subtype,
                             download=False, root=root, chunks=chunks,
                             contiguous=contiguous)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._root, download)

        if self._root_file is None:
            self._root_file = os.path.abspath(pjoin(self._root,
                                                    filename))
