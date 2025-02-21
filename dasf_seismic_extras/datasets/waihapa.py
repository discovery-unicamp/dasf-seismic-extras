#!/usr/bin/env python3

from dasf.datasets.download import DownloadWget

from dasf_seismic_extras.datasets import DatasetSeismicType
from dasf_seismic_extras.datasets.base import DatasetSEGY


class Waihapa(DatasetSEGY, DownloadWget):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        url = ("https://s3.amazonaws.com/open.source.geoscience/open_data/"
               "newzealand/Taranaiki_Basin/WAIHAPA-3D/3D-Waihapa.sgy")
        name = "Waihapa 3-D"
        filename = "waihapa.sgy"
        subtype = DatasetSeismicType.migrated_volume

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadWget.__init__(self, url, filename, self._root, download)
