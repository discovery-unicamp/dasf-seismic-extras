#!/usr/bin/env python3

from dasf.datasets.download import DownloadGDrive

from dasf_seismic_extras.datasets.base import DatasetSEGY, DatasetSeismicType


class Kahu(DatasetSEGY, DownloadGDrive):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Kahu"
        subtype = DatasetSeismicType.migrated_volume
        google_file_id = "1SOjbUHfz7_SO2vbq7jKdByjiLhK9kTf-"
        filename = "kahu.sgy"

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadGDrive.__init__(self, google_file_id, filename,
                                self._root, download)
