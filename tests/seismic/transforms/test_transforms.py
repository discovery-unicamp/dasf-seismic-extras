#!/usr/bin/env python3

import os
import shutil
import tempfile
import unittest

import numpy as np
import segyio
from dasf.datasets import DatasetArray, DatasetHDF5, DatasetZarr
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import (
    is_cpu_array,
    is_dask_cpu_array,
    is_dask_gpu_array,
    is_gpu_array,
)
from dasf_seismic_extras.datasets import DatasetSEGY
from dasf_seismic_extras.transforms import (
    ArrayToSEGY,
    SEGYToArray,
    SEGYToHDF5,
    SEGYToZarr,
    ZarrToSEGY,
)


class TestSEGYTo(unittest.TestCase):
    def setUp(self):
        self.segyfile = f"{tempfile.gettempdir()}/data.sgy"

        spec = segyio.spec()
        spec.format = 5
        spec.sorting = 2
        spec.samples = range(7)
        spec.ilines = range(1, 4)
        spec.xlines = range(1, 3)
        spec.offsets = range(1, 2)

        with segyio.create(self.segyfile, spec) as dst:
            arr = np.arange(start=0.000,
                            stop=0.007,
                            step=0.001,
                            dtype=np.single)

            arr = np.concatenate([[arr + 0.01], [arr + 0.02]], axis=0)
            lines = [arr + i for i in spec.ilines]
            cube = [(off * 100) + line for line in lines for off in spec.offsets]

            dst.iline[:, :] = cube

            for of in spec.offsets:
                for il in spec.ilines:
                    dst.header.iline[il, of] = {segyio.TraceField.INLINE_3D: il,
                                                segyio.TraceField.offset: of}

                for xl in spec.xlines:
                    dst.header.xline[xl, of] = {segyio.TraceField.CROSSLINE_3D: xl}

    @staticmethod
    def remove(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove dir and all contains
        else:
            raise ValueError("file {} is not a file or dir.".format(path))

    def tearDown(self):
        self.remove(self.segyfile)


class TestSEGYToArray(TestSEGYTo):
    def setUp(self):
        super().setUp()
        self.arrayfile = f"{tempfile.gettempdir()}/data.npy"

    def test_segy_to_array_cpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._load_cpu()

        T = SEGYToArray()

        T_1 = T.transform(dataset)

        self.assertTrue(isinstance(T_1, DatasetArray))
        self.assertTrue(is_cpu_array(T_1._load_cpu()._data))

        self.remove(self.arrayfile)

    def test_segy_to_array_mcpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._lazy_load_cpu()

        T = SEGYToArray()

        T_1 = T.transform(dataset)

        self.assertTrue(isinstance(T_1, DatasetArray))
        self.assertTrue(is_dask_cpu_array(T_1._lazy_load_cpu()._data))

        self.remove(self.arrayfile)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_segy_to_array_gpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._load_gpu()

        T = SEGYToArray()

        T_1 = T.transform(dataset)

        self.assertTrue(isinstance(T_1, DatasetArray))
        self.assertTrue(is_gpu_array(T_1._load_gpu()._data))

        self.remove(self.arrayfile)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_segy_to_array_mgpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._lazy_load_gpu()

        T = SEGYToArray()

        T_1 = T.transform(dataset)

        self.assertTrue(isinstance(T_1, DatasetArray))
        self.assertTrue(is_dask_gpu_array(T_1._lazy_load_gpu()._data))

        self.remove(self.arrayfile)


class TestSEGYToArrayToSEGY(TestSEGYTo):
    def setUp(self):
        super().setUp()
        self.arrayfile = f"{tempfile.gettempdir()}/data.npy"
        self.segyfile_2 = f"{tempfile.gettempdir()}/data_2.sgy"

    def test_segy_to_array_cpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._load_cpu()

        T_1 = SEGYToArray()
        T_2 = ArrayToSEGY(filename=self.segyfile_2)

        T_1_1 = T_1.transform(dataset)._load_cpu()
        T_2_1 = T_2.transform(T_1_1)

        self.assertTrue(isinstance(T_2_1, DatasetSEGY))

        self.remove(self.arrayfile)
        self.remove(self.segyfile_2)

    def test_segy_to_array_mcpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._lazy_load_cpu()

        T_1 = SEGYToArray()
        T_2 = ArrayToSEGY(filename=self.segyfile_2)

        T_1_1 = T_1.transform(dataset)._lazy_load_cpu()
        T_2_1 = T_2.transform(T_1_1)

        self.assertTrue(isinstance(T_2_1, DatasetSEGY))

        self.remove(self.arrayfile)
        self.remove(self.segyfile_2)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_segy_to_array_gpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._load_gpu()

        T_1 = SEGYToArray()
        T_2 = ArrayToSEGY(filename=self.segyfile_2)

        T_1_1 = T_1.transform(dataset)._load_gpu()
        T_2_1 = T_2.transform(T_1_1)

        self.assertTrue(isinstance(T_2_1, DatasetSEGY))

        self.remove(self.arrayfile)
        self.remove(self.segyfile_2)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_segy_to_array_mgpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._lazy_load_gpu()

        T_1 = SEGYToArray()
        T_2 = ArrayToSEGY(filename=self.segyfile_2)

        T_1_1 = T_1.transform(dataset)._lazy_load_gpu()
        print(T_1_1._data)
        T_2_1 = T_2.transform(T_1_1)

        self.assertTrue(isinstance(T_2_1, DatasetSEGY))

        self.remove(self.arrayfile)
        self.remove(self.segyfile_2)     


class TestSEGYToZarr(TestSEGYTo):
    def setUp(self):
        super().setUp()
        self.zarrfile = f"{tempfile.gettempdir()}/data.zarr"

    def test_segy_to_zarr_cpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._load_cpu()

        T = SEGYToZarr()

        T_1 = T._transform_cpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetZarr))

        self.remove(self.zarrfile)

    def test_segy_to_zarr_mcpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._lazy_load_cpu()

        T = SEGYToZarr()

        T_1 = T._lazy_transform_cpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetZarr))

        self.remove(self.zarrfile)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")        
    def test_segy_to_zarr_gpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._load_gpu()

        T = SEGYToZarr()

        T_1 = T._transform_gpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetZarr))

        self.remove(self.zarrfile)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_segy_to_zarr_mgpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False,
                              name="Test SEG-Y", chunks={"iline": 1})

        dataset = dataset._lazy_load_gpu()

        T = SEGYToZarr()

        T_1 = T._lazy_transform_gpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetZarr))

        self.remove(self.zarrfile)


class TestSEGYToHDF5(TestSEGYTo):
    def setUp(self):
        super().setUp()
        self.hdf5file = f"{tempfile.gettempdir()}/data.hdf5"

    def test_segy_to_hdf5_cpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False, name="Test SEG-Y")

        dataset = dataset._load_cpu()

        T = SEGYToHDF5(dataset_path="/dataset")

        T_1 = T._transform_cpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetHDF5))

        self.remove(self.hdf5file)

    def test_segy_to_hdf5_mcpu(self):
        dataset = DatasetSEGY(root=self.segyfile, download=False, name="Test SEG-Y")

        dataset = dataset._lazy_load_cpu()

        T = SEGYToHDF5(dataset_path="/dataset")

        T_1 = T._lazy_transform_cpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetHDF5))

        self.remove(self.hdf5file)
