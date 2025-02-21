#!/usr/bin/env python3

import os
from enum import Enum

import dask.array as da
import numpy as np
import numpy.lib.format
import segyio
from mdio import MDIOReader
from segysak import open_seisnc
from xarray import DataArray

try:
    import cupy as cp
except ImportError:
    pass

from dasf.datasets import Dataset
from dasf.utils.decorators import task_handler
from dasf.utils.funcs import human_readable_size

from dasf_seismic_extras.utils import utils as seismic_utils


class DatasetSeismicType(Enum):
    none = "none"
    cmp_gathers = "CMP Gathers"
    surface_seismic = "Surface Seismic"
    borehole_seismic = "Borehole Seismic"
    fourd_far_stack = "4D Far Stack"
    fourd_near_stack = "4D Near Stack"
    fourd_mid_stack = "4D Mid Stack"
    fourd_full_stack = "4D Full Stack"
    far_stack = "Far Stack"
    near_stack = "Near Stack"
    mid_stack = "Mid Stack"
    full_stack = "Full Stack"
    prestack_seismic = "Prestack Seismic"
    poststack_seismic = "Poststack Seismic"
    migrated_volume = "Migrated Volume"

    def __str__(self):
        return self.value


class DatasetSEGY(Dataset):
    def __init__(self,
                 name,
                 subtype=DatasetSeismicType.none,
                 download=True,
                 root=None,
                 chunks="auto",
                 contiguous=False,
                 *args,
                 **kwargs):

        Dataset.__init__(self, name, download, root)

        self._subtype = subtype

        self.chunks = chunks
        self.contiguous = contiguous

        self._root_file = root
        self.segyfile = None

        # Extra attributes for segyio
        self.args = args
        self.kwargs = kwargs

        if root is not None:
            if not os.path.isfile(root):
                raise Exception("SEG-Y requires a root=filename.")

            self._root = os.path.dirname(root)

            self._metadata = self._load_meta()

    def _download_check(self):
        """Load metadata after download.

        """
        self._metadata = self._load_meta()

    def _lazy_load(self, xp=np, **kwargs):
        # XXX: Warning, using float64 consumes 2x memory.
        dims = self.shape

        if self.contiguous:
            length = dims[0] * dims[1] * dims[2]
            blocks = da.zeros((length, 1), chunks=self.get_chunks(),
                              dtype=xp.float32)
        else:
            blocks = da.zeros(dims, chunks=self.get_chunks(),
                              dtype=xp.float32)

        blocks = blocks.map_blocks(seismic_utils.map_segy, self._root_file,
                                   self.contiguous, xp,
                                   *self.args, **self.kwargs,
                                   dtype=xp.float32, meta=xp.array(()))

        return blocks

    def _load(self, xp=np, **kwargs):
        # XXX: Warning, using float64 consumes 2x memory.
        local_data = xp.asarray(segyio.tools.cube(self.segyfile),
                                dtype=xp.float32)

        if self.contiguous:
            local_data = xp.ascontiguousarray(local_data)

        return local_data

    def _load_meta(self):
        assert self._root_file is not None, ("There is no temporary file to "
                                             "inspect")
        assert os.path.isfile(self._root_file), ("The root variable should "
                                                 "be a SEGY file")

        self.segyfile = segyio.open(self._root_file,
                                    *self.args, **self.kwargs)

        self.__parse_chunks()

        return self.inspect_segy_seismic_cube()

    def _lazy_load_gpu(self):
        self._data = self._lazy_load(cp)
        return self

    def _lazy_load_cpu(self):
        self._data = self._lazy_load(np)
        return self

    def _load_gpu(self):
        self._data = self._load(cp)
        return self

    def _load_cpu(self):
        self._data = self._load(np)
        return self

    @task_handler
    def load(self):
        ...

    def inspect_segy_seismic_cube(self):
        segy_file = self.segyfile

        segy_file_size = \
            human_readable_size(os.path.getsize(self._root_file),
                                decimal=2)

        iline_start = int(segy_file.ilines[0])
        iline_end = int(segy_file.ilines[-1])
        iline_offset = int(segy_file.ilines[1] - segy_file.ilines[0])

        xline_start = int(segy_file.xlines[0])
        xline_end = int(segy_file.xlines[-1])
        xline_offset = int(segy_file.xlines[1] - segy_file.xlines[0])

        time_start = int(segy_file.samples[0])
        time_end = int(segy_file.samples[-1])
        time_offset = int(segy_file.samples[1] - segy_file.samples[0])

        if self.chunks != "auto":
            if len(self.chunks) == 1:
                chunks_type = list(self.chunks.keys())[0]
            else:
                chunks_type = "block"
        else:
            chunks_type = "auto"

        segy_shape = (int((iline_end - iline_start)/iline_offset),
                      int((xline_end - xline_start)/xline_offset),
                      int((time_end - time_start)/time_offset))

        return {
            'size': segy_file_size,
            'file': self._root_file,
            'subtype': str(self._subtype),
            'shape': segy_shape,
            'iline': (iline_start, iline_end, iline_offset),
            'xline': (xline_start, xline_end, xline_offset),
            'z': (time_start, time_end, time_offset),
            'block': {
               "type": chunks_type,
               "chunks": self.chunks
            }
        }

    @property
    def shape(self):
        if self.segyfile is None:
            return 0

        ilsort = (self.segyfile.sorting ==
                  segyio.TraceSortingFormat.INLINE_SORTING)
        fast = self.segyfile.ilines if ilsort else self.segyfile.xlines
        slow = self.segyfile.xlines if ilsort else self.segyfile.ilines
        fast, slow, offs = len(fast), len(slow), len(self.segyfile.offsets)
        smps = len(self.segyfile.samples)
        return (fast, slow, smps) if offs == 1 else (fast, slow, offs, smps)

    @property
    def dtype(self):
        if self._data is not None:
            return self._data.dtype
        return None

    @property
    def ndim(self):
        if self._data is not None:
            return self._data.ndim
        return None

    def __parse_chunks(self):
        assert isinstance(self.chunks, dict) or self.chunks == "auto"

        if self.chunks == "auto":
            self.__chunks = self.chunks
            return

        dims = self.shape

        chunks = []
        if "iline" in self.chunks:
            chunks.append(self.chunks["iline"])
        else:
            chunks.append(dims[0])

        if "xline" in self.chunks:
            chunks.append(self.chunks["xline"])
        else:
            chunks.append(dims[1])

        if "twt" in self.chunks:
            chunks.append(self.chunks["twt"])
        else:
            chunks.append(dims[2])

        self.__chunks = tuple(chunks)

    def get_chunks(self):
        return self.__chunks

    def set_chunks(self, chunks):
        self.chunks = chunks

        self.__parse_chunks()

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def get_iline(self, idx):
        if self._data is not None:
            return self._data[idx, :, :]
        return None

    def get_xline(self, idx):
        if self._data is not None:
            return self._data[:, idx, :]
        return None

    def get_slice(self, idx):
        if self._data is not None:
            return self._data[:, :, idx]
        return None


class DatasetSEISNC(Dataset):
    def __init__(self,
                 name,
                 subtype=DatasetSeismicType.none,
                 download=True,
                 root=None,
                 chunks=None,
                 slicing="iline"):

        Dataset.__init__(self, name, download, root)

        self._subtype = subtype

        self.__chunks = chunks

        self._root_file = root

        if chunks and not isinstance(chunks, dict):
            raise Exception("Chunks should be a dict with format {'iline': x, "
                            "'xline': y, 'twt': z}.")

        if 'iline' not in chunks and \
           'xline' not in chunks and \
           'twt' not in chunks:
            raise Exception("Chunks should be a dict with format {'iline': x, "
                            "'xline': y, 'twt': z}.")

        if slicing != 'iline' and \
           slicing != 'xline' and \
           slicing != 'twt':
            raise Exception("Slices should be in ilines, xlines or twt.")

        self._slicing = slicing

    def _load_meta(self):
        assert self._root_file is not None, ("There is no temporary file to "
                                             "inspect")
        assert os.path.isfile(self._root_file), ("The root variable should "
                                                 "be a SEISNC file")

        return self.inspect_seisnc_seismic_cube()

    def _lazy_load_gpu(self):
        self._metadata = self._load_meta()
        self._data = open_seisnc(self._root_file, chunks=self.__chunks)
        if isinstance(self._data.data, DataArray):
            self._data.data.data = self._data.data.data.map_blocks(cp.asarray)
        else:
            self._data.data = self._data.data.map_blocks(cp.asarray)
        return self

    def _lazy_load_cpu(self):
        self._metadata = self._load_meta()
        self._data = open_seisnc(self._root_file, chunks=self.__chunks)
        return self

    def _load_gpu(self):
        return self._lazy_load_gpu()

    def _load_cpu(self):
        return self._lazy_load_cpu()

    @task_handler
    def load(self):
        ...

    def inspect_seisnc_seismic_cube(self):
        data = open_seisnc(self._root_file, chunks=self.__chunks)

        return {
            'size': data.seis.humanbytes,
            'file': self._root_file,
            'subtype': str(self._subtype),
            'shape': data.data.shape,
            'block': {
               "chunks": self.__chunks
            }
        }

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise IndexError("Index is 1-D.")

        iline_0 = 0
        iline_delta = 1
        if 'iline' in self._data.coords:
            iline_0 = self._data.coords['iline'][0]
            iline_delta = self._data.coords['iline'][1] - self._data.coords['iline'][0]

        xline_0 = 0
        xline_delta = 1
        if 'xline' in self._data.coords:
            xline_0 = self._data.coords['xline'][0]
            xline_delta = self._data.coords['xline'][1] - self._data.coords['xline'][0]

        twt_0 = 0
        twt_delta = 1
        if 'twt' in self._data.coords:
            twt_0 = self._data.coords['twt'][0]
            twt_delta = self._data.coords['twt'][1] - self._data.coords['twt'][0]

        if self._slicing == "iline":
            return self.get_iline(iline_0 + idx * iline_delta)
        elif self._slicing == "xline":
            return self.get_xline(xline_0 + idx * xline_delta)
        elif self._slicing == "twt":
            return self.get_slice(twt_0 + idx * twt_delta)

    def get_iline(self, idx):
        if self._data is not None:
            return self._data.sel(iline=(idx)) \
                                     .transpose("twt", "xline").data
        return None

    def get_xline(self, idx):
        if self._data is not None:
            return self._data.sel(xline=(idx)) \
                                     .transpose("twt", "iline").data
        return None

    def get_slice(self, idx):
        if self._data is not None:
            return self._data.sel(twt=(idx), method="nearest") \
                                     .transpose("iline", "xline").data


class DatasetMDIO(Dataset):
    def __init__(self,
                 name,
                 subtype=DatasetSeismicType.none,
                 download=True,
                 root=None,
                 chunks=None,
                 return_metadata=False):

        Dataset.__init__(self, name, download, root)

        self._subtype = subtype
        self._root_file = root

        self.chunks = chunks

        self.__return_metadata = return_metadata

        access_pattern = "012" # default access pattern
        if chunks:
            if not isinstance(chunks, dict):
                raise Exception("Chunks should be a dict with format {'iline': x, "
                                "'xline': y, 'twt': z}.")
            else:
                chunks_fields = ["iline", "xline", "twt"]
                access_pattern = ""
                for i, field in enumerate(chunks_fields):
                    if chunks.get(field) != -1:
                        access_pattern += str(i)
        self._access_pattern = access_pattern
                
        
        


        if root is not None:
            if not os.path.isdir(root):
                raise Exception("MDIO requires a root=filename.")

            self._root = os.path.dirname(root)

            self._metadata = self._load_meta()

    def _load_meta(self, backend="zarr"):
        assert self._root_file is not None, ("There is no temporary file to "
                                             "inspect")
        assert os.path.isdir(self._root_file), ("The root variable should "
                                                "be a MDIO file")

        self.__parse_chunks()

        return self.inspect_mdio_seismic_cube(backend=backend)

    def __read_mdio(self, backend="zarr"):
        return MDIOReader(self._root_file, backend=backend, new_chunks=self.__chunks,
                                access_pattern=self._access_pattern, return_metadata=self.__return_metadata)

    def _lazy_load_gpu(self):
        backend = "dask"
        self._metadata = self._load_meta(backend=backend)
        self._mdio = self.__read_mdio(backend=backend)
        self._data = self._mdio._traces.map_blocks(cp.asarray)
        return self

    def _lazy_load_cpu(self):
        backend = "dask"
        self._metadata = self._load_meta(backend=backend)
        self._mdio = self.__read_mdio(backend=backend)
        self._data = self._mdio._traces
        return self

    def _load_gpu(self):
        self._metadata = self._load_meta()
        self._mdio = self.__read_mdio()
        self._data = cp.asarray(self._mdio._traces)
        return self

    def _load_cpu(self):
        self._metadata = self._load_meta()
        self._mdio = self.__read_mdio()
        self._data = self._mdio._traces
        return self

    @task_handler
    def load(self):
        ...

    def inspect_mdio_seismic_cube(self, backend="zarr"):
        mdio = self.__read_mdio(backend=backend)

        mdio_size = 0
        if backend == "zarr":
            for z, v in mdio._traces.info_items():
                if z == "No. bytes":
                    mdio_size = int(v.split(' ')[0])
        elif backend == "dask":
            mdio_size = mdio._traces.nbytes
        else:
            raise ValueError(f"No valid {backend}.")

        return {
            'size': human_readable_size(mdio_size),
            'file': self._root_file,
            'subtype': str(self._subtype),
            'shape': mdio.shape,
            'samples': mdio.binary_header["Samples"],
            'interval': mdio.binary_header["Interval"],
            'block': {
               "chunks": self.__chunks
            }
        }
    
    def copy(self, url, **kwargs):
        if not hasattr(self, "_mdio"):
            raise Exception("Dataset must be loaded to be copied")
        self._mdio.copy(url, **kwargs)

    def __getitem__(self, idx):
        return self._data[idx]

    def get_iline(self, idx):
        if self._data is not None:
            return self._data.sel(iline=(idx)) \
                                     .transpose("twt", "xline").data
        return None

    def get_xline(self, idx):
        if self._data is not None:
            return self._data.sel(xline=(idx)) \
                                     .transpose("twt", "iline").data
        return None

    def get_slice(self, idx):
        if self._data is not None:
            return self._data.sel(twt=(idx), method="nearest") \
                                     .transpose("iline", "xline").data

    def __parse_chunks(self):
        assert self.chunks is None or isinstance(self.chunks, dict) or \
               self.chunks == "auto"

        if self.chunks is None or self.chunks == "auto":
            self.__chunks = self.chunks
            return

        dims = (64, 64, 64) # MDIO default chunking

        chunks = []
        if "iline" in self.chunks:
            chunks.append(self.chunks["iline"])
        else:
            chunks.append(dims[0])

        if "xline" in self.chunks:
            chunks.append(self.chunks["xline"])
        else:
            chunks.append(dims[1])

        if "twt" in self.chunks:
            chunks.append(self.chunks["twt"])
        else:
            chunks.append(dims[2])

        self.__chunks = tuple(chunks)
