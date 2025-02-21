"""V2 validation test suite"""
import csv
import os
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
from pytest import fixture
from utils import get_class_name, get_item


@parameterized_class(
    complex_trace
    + dip_azm
    + edge_detection
    + frequency
    + noise_reduction
    + signal,
    class_name_func=get_class_name,
)
class TestV2(unittest.TestCase):
    """Base class for V2 validation tests"""

    # Default Test params
    operator_params = {}
    v2_in_shape_chunks = (10, 10, -1)
    v2_valid_precision = -15
    v2_preprocessing = None
    v2_slice = None
    v2_precision=5
    v2_outer_dim = None
    v2_skip = False

    @fixture(autouse=True)
    def input_data(self, request):
        filename = request.module.__file__
        test_dir = os.path.split(filename)[:-1]
        self.test_dir = os.path.join(*test_dir)
        raw_path = os.path.join(
            self.test_dir, "test_validation", "validation", "Raw.npy"
        )
        self.in_data = np.load(raw_path).astype("float64")

    def test_lazy_implementation_cpu(self):
        """Test result values of CPU lazy implementation using CPU as ground truth"""
        operator = self.operator_cls(**self.operator_params)

        if self.v2_skip:
            raise self.skipTest(f"{operator.__class__.__name__}: SKIP")

        if self.v2_slice:
            in_data = [self.in_data[:self.v2_slice[0],:self.v2_slice[1],:self.v2_slice[2]]]
        else:
            in_data = [self.in_data]
        if self.v2_outer_dim:
            in_data = [np.broadcast_to(in_data[0], (self.v2_outer_dim,) + in_data[0].shape)]
        if self.v2_preprocessing:
            for stage in self.v2_preprocessing:
                outputs = stage[0]()._transform_cpu(*in_data)
                in_data = []
                for index in stage[1]:
                    in_data.append(outputs[index])

        in_data_cpu = in_data
        in_data_lazy_cpu = [
            da.from_array(data, chunks=get_item(self.v2_in_shape_chunks, i))
            for i, data in enumerate(in_data_cpu)
        ]
        try:
            out_data_cpu = operator._transform_cpu(*in_data_cpu)
            out_data_lazy_cpu = operator._lazy_transform_cpu(*in_data_lazy_cpu)
        except NotImplementedError as nie:
            raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

        out_data_cpu = (
            out_data_cpu if isinstance(out_data_cpu, tuple) else (out_data_cpu,)
        )
        out_data_lazy_cpu = (
            out_data_lazy_cpu
            if isinstance(out_data_lazy_cpu, tuple)
            else (out_data_lazy_cpu,)
        )

        self.assertEqual(
            len(out_data_cpu),
            len(out_data_lazy_cpu),
            "Number of output arrays check",
        )
        for i, outputs in enumerate(zip(out_data_cpu, out_data_lazy_cpu)):
            output_comp = outputs[1].compute()
            np.testing.assert_almost_equal(
                outputs[0],
                output_comp,
                decimal=self.v2_precision,
                err_msg=f"Output {i} - CPU x Dask CPU Comparison",
            )

    @unittest.skipIf(not is_gpu_supported(), "not supported CUDA in this platform")
    def test_implementation_gpu(self):
        """Test result values of GPU implementation using CPU as ground truth"""
        operator = self.operator_cls(**self.operator_params)
        dtypes = ["float32", "float64"]
        input_data = {"cpu": {}, "gpu": {}}
        output_data = {"cpu": {}, "gpu": {}}

        if self.v2_skip:
            raise self.skipTest(f"{operator.__class__.__name__}: SKIP")
        
        if self.v2_slice:
            in_data = [self.in_data[:self.v2_slice[0],:self.v2_slice[1],:self.v2_slice[2]]]
        else:
            in_data = [self.in_data]
        if self.v2_outer_dim:
            in_data = [np.broadcast_to(in_data[0], (self.v2_outer_dim,) + in_data[0].shape)]
        if self.v2_preprocessing:
            for stage in self.v2_preprocessing:
                outputs = stage[0]()._transform_cpu(*in_data)
                in_data = []
                for index in stage[1]:
                    in_data.append(outputs[index])

        for dtype in dtypes:
            input_data["cpu"][dtype] = [data.astype(dtype) for data in in_data]
            input_data["gpu"][dtype] = [
                cp.array(data) for data in input_data["cpu"][dtype]
            ]

        try:
            for dtype in dtypes:
                output_data["cpu"][dtype] = operator._transform_cpu(
                    *input_data["cpu"][dtype]
                )
                output_data["gpu"][dtype] = operator._transform_gpu(
                    *input_data["gpu"][dtype]
                )

        except NotImplementedError as nie:
            raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

        for arch in output_data:
            for dtype in output_data[arch]:
                output_data[arch][dtype] = (
                    output_data[arch][dtype]
                    if isinstance(output_data[arch][dtype], tuple)
                    else (output_data[arch][dtype],)
                )

        self.assertEqual(
            len(output_data["cpu"]["float64"]),
            len(output_data["gpu"]["float64"]),
            "Number of output arrays check",
        )
        for i in range(len(output_data["cpu"]["float64"])):
            errors = {}
            for dtype in dtypes:
                gpu_out = output_data["gpu"][dtype][i].get()
                cpu_out = output_data["cpu"][dtype][i]
                inf_value = int(np.max(np.abs(cpu_out)[np.abs(cpu_out) != np.inf]) * 1_000_000)
                gpu_out[gpu_out == np.inf] = inf_value
                gpu_out[gpu_out == -np.inf] = -inf_value
                cpu_out[cpu_out == np.inf] = inf_value
                cpu_out[cpu_out == -np.inf] = -inf_value
                errors[dtype] = np.mean(
                    np.abs(gpu_out - cpu_out)
                )

            if errors["float32"] == 0 and errors["float64"] == 0:
                with open("operators.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([operator.__class__.__name__, "ERR0", "ERR0"])
                continue

            errors["float32"] = np.log10(errors["float32"])
            errors["float64"] = np.log10(errors["float64"])
            with open("operators.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([operator.__class__.__name__, errors['float32'], errors['float64']])

            if errors["float32"] < self.v2_valid_precision and errors["float64"] < self.v2_valid_precision:
                continue
            if errors["float32"] > 0 and errors["float64"] > 0:
                factor = errors["float32"] / errors["float64"]
            elif errors["float32"] < 0 and errors["float64"] < 0:
                factor = errors["float64"] / errors["float32"]
            elif errors["float32"] >= 0:
                factor = np.abs(errors["float64"] / errors["float32"]) + 1
            else:
                factor = -1  # Float 64 error is higher than Float 32

            self.assertGreaterEqual(
                factor,
                1.9,
                f"Output {i} - Check float precision error. F32 mean: {errors['float32']}. F64 mean: {errors['float64']}",
            )

    @unittest.skipIf(not is_gpu_supported(), "not supported CUDA in this platform")
    def test_lazy_implementation_gpu(self):
        """Test result values of GPU lazy implementation using GPU as ground truth"""
        operator = self.operator_cls(**self.operator_params)
        if self.v2_skip:
            raise self.skipTest(f"{operator.__class__.__name__}: SKIP")

        if self.v2_slice:
            in_data = [cp.array(self.in_data[:self.v2_slice[0],:self.v2_slice[1],:self.v2_slice[2]])]
        else:
            in_data = [cp.array(self.in_data)]
        if self.v2_outer_dim:
            in_data = [cp.broadcast_to(in_data[0], (self.v2_outer_dim,) + in_data[0].shape)]
        if self.v2_preprocessing:
            for stage in self.v2_preprocessing:
                outputs = stage[0]()._transform_gpu(*in_data)
                in_data = []
                for index in stage[1]:
                    in_data.append(outputs[index])

        in_data_gpu = in_data
        in_data_lazy_gpu = [
            da.from_array(data, chunks=get_item(self.v2_in_shape_chunks, i))
            for i, data in enumerate(in_data_gpu)
        ]
        try:
            out_data_gpu = operator._transform_gpu(*in_data_gpu)
            out_data_lazy_gpu = operator._lazy_transform_gpu(*in_data_lazy_gpu)
        except NotImplementedError as nie:
            raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

        out_data_gpu = (
            out_data_gpu if isinstance(out_data_gpu, tuple) else (out_data_gpu,)
        )
        out_data_lazy_gpu = (
            out_data_lazy_gpu
            if isinstance(out_data_lazy_gpu, tuple)
            else (out_data_lazy_gpu,)
        )

        self.assertEqual(
            len(out_data_gpu),
            len(out_data_lazy_gpu),
            "Number of output arrays check",
        )
        for i, outputs in enumerate(zip(out_data_gpu, out_data_lazy_gpu)):
            # some functions may not be implemented on the GPU lib, and they are only invoked in compute
            try:
                output_comp = outputs[1].compute()
            except NotImplementedError as nie:
                raise self.skipTest(f"{operator.__class__.__name__}: {str(nie)}")

            np.testing.assert_almost_equal(
                outputs[0].get(),
                output_comp.get(),
                decimal=self.v2_precision,
                err_msg=f"Output {i} - GPU x Dask GPU Comparison",
            )
