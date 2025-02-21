#!/usr/bin/env python3

import os
import unittest

from dasf.utils.funcs import is_gpu_supported
from parameterized import parameterized_class
from pytest import fixture

from dasf_seismic_extras.datasets import DatasetSEGY, DatasetSEISNC


def parameterize_dataset_type():
    datasets = [
        {"name": "SEGY", "cls": "DatasetSEGY", "file": "Array.sgy", "extra_args": {}},
        {"name": "SEISNC", "cls": "DatasetSEISNC", "file": "Array.seisnc", "extra_args": {"chunks": {"iline": 10, "xline": 10, "twt": 10}}},
    ]
    
    return datasets
    

@parameterized_class(parameterize_dataset_type())
class TestTypes(unittest.TestCase):
    @fixture(autouse=True)
    def data_dir(self, request):
        filename = request.module.__file__
        self.test_dir, _ = os.path.splitext(filename)
        
    def test_dataset_load(self):
        raw_path = os.path.join(self.test_dir, "simple",
                                self.file)
                                
        dataset = eval(self.cls)(name=self.name, root=raw_path, download=False, **self.extra_args)
        dataset.load()

        self.assertTrue(hasattr(dataset, '_metadata'))
        self.assertTrue("size" in dataset._metadata)
