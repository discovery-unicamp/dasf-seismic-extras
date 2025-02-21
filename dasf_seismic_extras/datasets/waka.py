#!/usr/bin/env python3

import os
from os.path import join as pjoin

import requests

from dasf_seismic_extras.datasets.base import DatasetSEGY, DatasetSeismicType


class Waka(DatasetSEGY):
    def __init__(self,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False):

        name = "Waka 3-D"
        subtype = DatasetSeismicType.prestack_seismic

        DatasetSEGY.__init__(self, name=name, subtype=subtype,
                             download=download, root=root,
                             chunks=chunks, contiguous=contiguous)

    def get_filename_path(self):
        if self._root is None or os.path.isdir(self._root):
            filename = "waka.py"

            return os.path.abspath(pjoin(self._root, filename))
        return self._root

    def download(self):
        URL = ("http://s3.amazonaws.com/open.source.geoscience/open_data/"
               "newzealand/Cantebury_Basin/WAKA-3D/"
               "BO_WAKA3D-PR3988-FS_3D_Final_Stack.sgy.parta")

        parts = ['a', 'b', 'c', 'd', 'e']

        for part in parts:
            response = requests.get(URL + part, stream=True)

            with open(self.get_filename_path(), "ab") as handle:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        handle.write(chunk)
