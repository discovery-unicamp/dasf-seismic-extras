"""V1 validation test suite"""
import unittest

import dask.array as da
import numpy as np

try:
    import cupy as cp
except Exception:
    pass

from dasf.utils.funcs import is_gpu_supported
from parameterized import parameterized_class
from params_complex_trace import attributes as complex_trace
from params_dip_azm import attributes as dip_azm
from params_edge_detection import attributes as edge_detection
from params_frequency import attributes as frequency
from params_noise_reduction import attributes as noise_reduction
from params_signal import attributes as signal
from utils import generate_input_data, get_class_name, get_item


@parameterized_class(
    complex_trace
    + dip_azm
    + edge_detection
    + frequency
    + noise_reduction
    + signal,
    class_name_func=get_class_name,
)
class TestV1(unittest.TestCase):
    """Base class for V1 validation tests"""

    # Default Test params
    dtypes = {"float32": "float32", "float64": "float64"}
    operator_params = {}
    v1_in_shape = (20, 20, 20)
    v1_in_shape_chunks = (5, 5, 5)
    v1_expected_shape = None
    v1_inputs = 1
    v1_outputs = 1

    def test_shape_dtype_from_cpu(self):
        """Test shape and dtype of operator's CPU implementation"""
        operator = self.operator_cls(**self.operator_params)

        for in_dtype, expected_dtype in self.dtypes.items():
            in_data = generate_input_data(
                self.v1_inputs, self.v1_in_shape, in_dtype, np
            )
            try:
                out_data = operator._transform_cpu(*in_data)
            except NotImplementedError as nie:
                raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

            if not isinstance(out_data, tuple):
                out_data = (out_data,)

            self.assertEqual(
                self.v1_outputs,
                len(out_data),
                f"Number of output arrays check for {in_dtype}",
            )
            for i, output in enumerate(out_data):
                self.assertEqual(
                    get_item(self.v1_expected_shape, i)
                    if self.v1_expected_shape
                    else get_item(self.v1_in_shape, 0),
                    output.shape,
                    f"Output {i} - Shape check for {in_dtype}",
                )
                self.assertEqual(
                    get_item(expected_dtype, i),
                    output.dtype,
                    f"Output {i} - Data Type check for {in_dtype}",
                )

    def test_shape_dtype_from_lazy_cpu(self):
        """Test shape and dtype of operator's Lazy CPU implementation"""
        operator = self.operator_cls(**self.operator_params)

        for in_dtype, expected_dtype in self.dtypes.items():
            in_data = generate_input_data(
                self.v1_inputs, self.v1_in_shape, in_dtype, np
            )
            # The input data is small, so we can import from array
            in_data = [
                da.from_array(data, chunks=get_item(self.v1_in_shape_chunks, i))
                for i, data in enumerate(in_data)
            ]

            try:
                out_data = operator._lazy_transform_cpu(*in_data)
            except NotImplementedError as nie:
                raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

            if not isinstance(out_data, tuple):
                out_data = (out_data,)

            self.assertEqual(
                self.v1_outputs,
                len(out_data),
                f"Number of output arrays check for {in_dtype}",
            )
            for i, output in enumerate(out_data):
                output_comp = output.compute()
                self.assertEqual(
                    get_item(self.v1_expected_shape, i)
                    if self.v1_expected_shape
                    else get_item(self.v1_in_shape, 0),
                    output_comp.shape,
                    f"Output {i} - Shape check for {in_dtype}",
                )
                self.assertEqual(
                    get_item(expected_dtype, i),
                    output_comp.dtype,
                    f"Output {i} - Data Type check for {in_dtype}",
                )
                self.assertEqual(
                    np.ndarray,
                    type(output_comp),
                    f"Output {i} - Computed Array type check for {in_dtype}",
                )
                self.assertEqual(
                    np.ndarray,
                    type(output._meta),
                    f"Output {i} - Dask Array meta type check for {in_dtype}",
                )

    @unittest.skipIf(not is_gpu_supported(), "not supported CUDA in this platform")
    def test_shape_dtype_from_gpu(self):
        """Test shape and dtype of operator's GPU implementation"""
        operator = self.operator_cls(**self.operator_params)

        for in_dtype, expected_dtype in self.dtypes.items():
            in_data = generate_input_data(
                self.v1_inputs, self.v1_in_shape, in_dtype, cp
            )
            try:
                out_data = operator._transform_gpu(*in_data)
            except NotImplementedError as nie:
                raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

            if not isinstance(out_data, tuple):
                out_data = (out_data,)

            self.assertEqual(
                self.v1_outputs,
                len(out_data),
                f"Number of output arrays check for {in_dtype}",
            )
            for i, output in enumerate(out_data):
                self.assertEqual(
                    get_item(self.v1_expected_shape, i)
                    if self.v1_expected_shape
                    else get_item(self.v1_in_shape, 0),
                    output.shape,
                    f"Output {i} - Shape check for {in_dtype}",
                )
                self.assertEqual(
                    get_item(expected_dtype, i),
                    output.dtype,
                    f"Output {i} - Data Type check for {in_dtype}",
                )

    @unittest.skipIf(not is_gpu_supported(), "not supported CUDA in this platform")
    def test_shape_dtype_from_lazy_gpu(self):
        """Test shape and dtype of operator's Lazy GPU implementation"""
        operator = self.operator_cls(**self.operator_params)

        for in_dtype, expected_dtype in self.dtypes.items():
            in_data = generate_input_data(
                self.v1_inputs, self.v1_in_shape, in_dtype, cp
            )
            # The input data is small, so we can import from array
            in_data = [
                da.from_array(data, chunks=get_item(self.v1_in_shape_chunks, i))
                for i, data in enumerate(in_data)
            ]

            try:
                out_data = operator._lazy_transform_gpu(*in_data)
            except NotImplementedError as nie:
                raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

            if not isinstance(out_data, tuple):
                out_data = (out_data,)

            self.assertEqual(
                self.v1_outputs,
                len(out_data),
                f"Number of output arrays check for {in_dtype}",
            )
            for i, output in enumerate(out_data):
                # some functions may not be implemented on the GPU lib, and they are only invoked in compute
                try:
                    output_comp = output.compute()
                except NotImplementedError as nie:
                    raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")
                self.assertEqual(
                    get_item(self.v1_expected_shape, i)
                    if self.v1_expected_shape
                    else get_item(self.v1_in_shape, 0),
                    output_comp.shape,
                    f"Output {i} - Shape check for {in_dtype}",
                )
                self.assertEqual(
                    get_item(expected_dtype, i),
                    output_comp.dtype,
                    f"Output {i} - Data Type check for {in_dtype}",
                )
                self.assertEqual(
                    cp.ndarray,
                    type(output_comp),
                    f"Output {i} - Computed Array type check for {in_dtype}",
                )
                self.assertEqual(
                    cp.ndarray,
                    type(output._meta),
                    f"Output {i} - Dask Array meta type check for {in_dtype}",
                )
