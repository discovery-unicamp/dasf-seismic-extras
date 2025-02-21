#!/usr/bin/env python3

from dasf.datasets.download import DownloadWget

from dasf_seismic_extras.datasets.base import DatasetSEGY, DatasetSeismicType


class Opunake(DatasetSEGY, DownloadWget):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Opunake 3-D"
        subtype = DatasetSeismicType.prestack_seismic
        url = ("http://s3.amazonaws.com/open.source.geoscience/open_data/"
               "newzealand/Taranaiki_Basin/OPUNAKE-3D/"
               "OPUNAKE3D-PR3461-FS.3D.Final_Stack.sgy")
        filename = "opunake.sgy"

        DatasetSEGY.__init__(self, name=name, subtype=subtype, download=False,
                             root=root, chunks=chunks, contiguous=contiguous)

        DownloadWget.__init__(self, url, filename, self._root, download)
