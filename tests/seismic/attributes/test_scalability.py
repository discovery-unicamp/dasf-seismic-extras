""" Scalability Tests """

import configparser
import os
import unittest

try:
    import cupy as cp
except ImportError:
    pass

import dask.array as da
from distributed import Client
from parameterized import parameterized
from paramiko.client import SSHClient
from params_complex_trace import attributes as complex_trace
from params_dip_azm import attributes as dip_azm
from params_edge_detection import attributes as edge_detection
from params_frequency import attributes as frequency
from params_noise_reduction import attributes as noise_reduction
from params_signal import attributes as signal
from pytest import fixture
from utils import get_func_name


def parameterize_matrix():
    # Tested attributes
    operators = complex_trace \
                + dip_azm \
                + edge_detection \
                + frequency \
                + noise_reduction \
                + signal

    # Number of cores
    cores = [1, 2, 4]

    # Type of arches
    arches = ["cpu", "gpu"]

    params = []
    for operator in operators:
        for core in cores:
            for arch in arches:
                params.append((operator["operator_cls"],
                               core,
                               arch))

    return params


def run_command(username, server, cmdline):
    client = SSHClient()
    client.load_system_host_keys()
    client.connect(username=username, hostname=server)

    return client.exec_command(cmdline)


def setup_scheduler(username, scheduler, jsonfile, cmd_prefix=''):
    scheduler_cmd = f'{cmd_prefix} dask scheduler --scheduler-file {jsonfile}'

    return run_command(username, scheduler, scheduler_cmd)


def setup_workers(username, worker, jsonfile, is_gpu=False, cmd_prefix=''):
    if is_gpu:
        worker_cmd = f'{cmd_prefix} dask-cuda-worker --scheduler-file {jsonfile}'
    else:
        worker_cmd = f'{cmd_prefix} dask worker --scheduler-file {jsonfile}'

    return run_command(username, worker, worker_cmd)


class TestScalabilityAttributes(unittest.TestCase):
    # Default Test params
    operator_params = {}

    @fixture(autouse=True)
    def pre_setUp(self, request):
        self._skip = False

        try:
            config_file = os.environ['DASF_TEST_SCALABILITY_CONFIG']
            try:
                self._cmd_prefix = os.environ['DASF_TEST_SCALABILITY_CMD_PREFIX']
            except:
                self._cmd_prefix = ""

            self._config = configparser.ConfigParser()
            self._config.read(config_file)
        except Exception as e:
            self._skip = True

    @fixture(autouse=True)
    def setupBenchmark(self, benchmark):
        self.benchmark = benchmark

    @parameterized.expand(parameterize_matrix(), name_func=get_func_name)
    def test_scalability(self, operator_cls, core, arch):
        client = None

        try:
            username = self._config["test"]["username"]
            jsonfile = self._config["test"]["jsonfile"]

            setup_scheduler(username, self._config["test"]["scheduler"],
                            jsonfile, cmd_prefix=self._cmd_prefix)

            while not os.path.exists(jsonfile):
                continue

            nworkers = 0
            for worker in self._config["test"]["workers"].split(","):
                if nworkers == core:
                    break
                nworkers += 1

                setup_workers(username, worker, jsonfile,
                              is_gpu=(True if arch == "gpu" else False),
                              cmd_prefix=self._cmd_prefix)

                client = Client(scheduler_file=jsonfile)

                client.wait_for_workers(core)
        except Exception as e:
            raise self.skipTest(f"{operator_cls.__class__.__name__}: SKIP")

        if self._skip:
            raise self.skipTest(f"{operator_cls.__class__.__name__}: NO CONFIG FILE... SKIP")

        # * This input has about 1 GB
        # * The chunks should keep the same dimention of axis Z
        dataset = da.random.random((400, 450, 500), chunks=(100, 100, -1))

        operator = operator_cls(**self.operator_params)

        if arch == "gpu":
            def send_to_gpu(block):
                return cp.array(block)

            dataset = dataset.map_blocks(send_to_gpu)

            self.benchmark(operator._lazy_transform_gpu(X=dataset).compute)
        elif arch == "cpu":
            self.benchmark(operator._lazy_transform_cpu(X=dataset).compute)

        if jsonfile and client:
            client.shutdown()
