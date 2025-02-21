#!/usr/bin/env python3

import dask.array as da
import numpy as np
from numba import cuda, njit
from pkg_resources import parse_version
from scipy import signal

try:
    import cupy as cp
    if cp.cuda.runtime.runtimeGetVersion() >= 12000:
        import cupyx.scipy.signal as cusignal
    else:
        import cusignal
except ImportError:
    pass

from dasf.transforms import Transform
from dasf_seismic.utils.utils import dask_overlap, dask_trim_internal

from dasf_seismic_extras.attributes.signal import FirstDerivative
from dasf_seismic_extras.utils.utils import (
    dask_cupy_angle_wrapper,
    local_events,
    set_time_chunk_overlap,
)


class Hilbert(Transform):
    def __init__(self):
        super().__init__()

    def __real_signal_hilbert(self, X, xsignal):
        # Avoiding return complex128
        return xsignal.hilbert(X)

    def _lazy_transform(self, X, xsignal, xp):
        return X.map_blocks(self.__real_signal_hilbert, xsignal=xsignal,
                            dtype=X.dtype, meta=xp.array((), dtype=X.dtype))

    def _lazy_transform_cpu(self, X):
        kernel = set_time_chunk_overlap(X)

        if kernel:
            X = dask_overlap(X, kernel)

        analytical_trace = self._lazy_transform(X, xsignal=signal, xp=np)

        if kernel:
            return dask_trim_internal(analytical_trace, kernel)
        return analytical_trace

    def _lazy_transform_gpu(self, X):
        kernel = set_time_chunk_overlap(X)

        if kernel:
            X = dask_overlap(X, kernel)

        analytical_trace = self._lazy_transform(X, xsignal=cusignal, xp=cp)

        if kernel:
            return dask_trim_internal(analytical_trace, kernel)
        return analytical_trace

    def _transform_cpu(self, X):
        return self.__real_signal_hilbert(X, signal)

    def _transform_gpu(self, X):
        return self.__real_signal_hilbert(X, cusignal)


class Envelope(Hilbert):
    def _lazy_transform_cpu(self, X):
        analytical_trace = super()._lazy_transform_cpu(X)

        return da.absolute(analytical_trace)

    def _lazy_transform_gpu(self, X):
        analytical_trace = super()._lazy_transform_gpu(X)

        return da.absolute(analytical_trace)

    def _transform_cpu(self, X):
        return np.absolute(super()._transform_cpu(X))

    def _transform_gpu(self, X):
        return cp.absolute(super()._transform_gpu(X))


class InstantaneousPhase(Hilbert):
    def _lazy_transform_cpu(self, X):
        analytical_trace = super()._lazy_transform_cpu(X)

        return da.rad2deg(da.angle(analytical_trace))

    def _lazy_transform_gpu(self, X):
        analytical_trace = super()._lazy_transform_gpu(X)

        if parse_version(cp.__version__) < parse_version("12.0.0"):
            return da.rad2deg(dask_cupy_angle_wrapper(analytical_trace))
        return da.rad2deg(da.angle(analytical_trace))

    def _transform_cpu(self, X):
        analytical_trace = super()._transform_cpu(X)

        return np.rad2deg(np.angle(analytical_trace))

    def _transform_gpu(self, X):
        analytical_trace = super()._transform_gpu(X)

        return cp.rad2deg(cp.angle(analytical_trace))


class CosineInstantaneousPhase(Hilbert):
    def _lazy_transform_gpu(self, X):
        analytical_trace = super()._lazy_transform_gpu(X)

        if parse_version(cp.__version__) < parse_version("12.0.0"):
            return da.cos(dask_cupy_angle_wrapper(analytical_trace))
        return da.cos(da.angle(analytical_trace))

    def _lazy_transform_cpu(self, X):
        analytical_trace = super()._lazy_transform_cpu(X)

        return da.cos(da.angle(analytical_trace))

    def _transform_gpu(self, X):
        return cp.cos(cp.angle(super()._transform_gpu(X)))

    def _transform_cpu(self, X):
        return np.cos(np.angle(super()._transform_cpu(X)))


class RelativeAmplitudeChange(Transform):
    def __init__(self):
        super().__init__()

        self.__envelope = Envelope()
        self.__first_derivative = FirstDerivative(axis=-1)

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        env_prime = self.__first_derivative._lazy_transform_gpu(env)

        result = env_prime / env

        result[da.isnan(result)] = 0

        return da.clip(result, -1, 1)

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        env_prime = self.__first_derivative._lazy_transform_cpu(env)

        result = env_prime / env

        result[da.isnan(result)] = 0

        return da.clip(result, -1, 1)

    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        env_prime = self.__first_derivative._transform_gpu(env)

        result = env_prime / env

        result[da.isnan(result)] = 0

        return cp.clip(result, -1, 1)

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        env_prime = self.__first_derivative._transform_cpu(env)

        result = env_prime / env

        result[da.isnan(result)] = 0

        return np.clip(result, -1, 1)


class AmplitudeAcceleration(RelativeAmplitudeChange):
    def __init__(self):
        super().__init__()

        self.__first_derivative = FirstDerivative(axis=-1)

    def _lazy_transform_gpu(self, X):
        rac = super()._lazy_transform_gpu(X)

        return self.__first_derivative._lazy_transform_gpu(rac)

    def _lazy_transform_cpu(self, X):
        rac = super()._lazy_transform_cpu(X)

        return self.__first_derivative._lazy_transform_cpu(rac)

    def _transform_gpu(self, X):
        rac = super()._transform_gpu(X)

        return self.__first_derivative._transform_gpu(rac)

    def _transform_cpu(self, X):
        rac = super()._transform_cpu(X)

        return self.__first_derivative._transform_cpu(rac)


class InstantaneousFrequency(Transform):
    def __init__(self, sample_rate=4):
        super().__init__()

        self._sample_rate = sample_rate

        self.__inst_phase = InstantaneousPhase()
        self.__first_derivative = FirstDerivative(axis=-1)

    def _lazy_transform_gpu(self, X):
        fs = 1000 / self._sample_rate

        phase = self.__inst_phase._lazy_transform_gpu(X)
        phase = da.deg2rad(phase)
        phase = phase.map_blocks(cp.unwrap, dtype=X.dtype)

        phase_prime = self.__first_derivative._lazy_transform_gpu(phase)

        return da.absolute((phase_prime / (2.0 * np.pi) * fs))

    def _lazy_transform_cpu(self, X):
        fs = 1000 / self._sample_rate

        phase = self.__inst_phase._lazy_transform_cpu(X)
        phase = da.deg2rad(phase)
        phase = phase.map_blocks(np.unwrap, dtype=X.dtype)

        phase_prime = self.__first_derivative._lazy_transform_cpu(phase)

        return da.absolute((phase_prime / (2.0 * np.pi) * fs))

    def _transform_gpu(self, X):
        fs = 1000 / self._sample_rate

        phase = self.__inst_phase._transform_gpu(X)
        phase = cp.deg2rad(phase)
        phase = cp.unwrap(phase)

        phase_prime = self.__first_derivative._transform_gpu(phase)

        return cp.absolute((phase_prime / (2.0 * np.pi) * fs))

    def _transform_cpu(self, X):
        fs = 1000 / self._sample_rate

        phase = self.__inst_phase._transform_cpu(X)
        phase = np.deg2rad(phase)
        phase = np.unwrap(phase)

        phase_prime = self.__first_derivative._transform_cpu(phase)

        return np.absolute((phase_prime / (2.0 * np.pi) * fs))


class InstantaneousBandwidth(RelativeAmplitudeChange):
    def _lazy_transform_gpu(self, X):
        rac = super()._lazy_transform_gpu(X)

        return da.absolute(rac) / (2.0 * np.pi)

    def _lazy_transform_cpu(self, X):
        rac = super()._lazy_transform_cpu(X)

        return da.absolute(rac) / (2.0 * np.pi)

    def _transform_gpu(self, X):
        rac = super()._transform_gpu(X)

        return cp.absolute(rac) / (2.0 * np.pi)

    def _transform_cpu(self, X):
        rac = super()._transform_cpu(X)

        return np.absolute(rac) / (2.0 * np.pi)


class DominantFrequency(Transform):
    def __init__(self, sample_rate=4):
        super().__init__()

        self._sample_rate = sample_rate

        self.__inst_freq = InstantaneousFrequency(sample_rate=sample_rate)
        self.__inst_band = InstantaneousBandwidth()

    def _lazy_transform_gpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_gpu(X)
        inst_band = self.__inst_band._lazy_transform_gpu(X)
        return da.hypot(inst_freq, inst_band)

    def _lazy_transform_cpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_cpu(X)
        inst_band = self.__inst_band._lazy_transform_cpu(X)
        return da.hypot(inst_freq, inst_band)

    def _transform_gpu(self, X):
        inst_freq = self.__inst_freq._transform_gpu(X)
        inst_band = self.__inst_band._transform_gpu(X)
        return cp.hypot(inst_freq, inst_band)

    def _transform_cpu(self, X):
        inst_freq = self.__inst_freq._transform_cpu(X)
        inst_band = self.__inst_band._transform_cpu(X)
        return np.hypot(inst_freq, inst_band)


class FrequencyChange(Transform):
    def __init__(self, sample_rate=4):
        super().__init__()

        self._sample_rate = sample_rate

        self.__inst_freq = InstantaneousFrequency(sample_rate=sample_rate)
        self.__first_derivative = FirstDerivative(axis=-1)

    def _lazy_transform_gpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_gpu(X)
        return self.__first_derivative._lazy_transform_gpu(inst_freq)

    def _lazy_transform_cpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_cpu(X)
        return self.__first_derivative._lazy_transform_cpu(inst_freq)

    def _transform_gpu(self, X):
        inst_freq = self.__inst_freq._transform_gpu(X)
        return self.__first_derivative._transform_gpu(inst_freq)

    def _transform_cpu(self, X):
        inst_freq = self.__inst_freq._transform_cpu(X)
        return self.__first_derivative._transform_cpu(inst_freq)


class Sweetness(Envelope):
    def __init__(self, sample_rate=4):
        super().__init__()

        self._sample_rate = sample_rate

        self.__inst_freq = InstantaneousFrequency(sample_rate=sample_rate)

    def __sweetness_limit(self, X):
        X[X < 5] = 5
        return X

    def _lazy_transform_gpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_gpu(X)
        inst_freq = inst_freq.map_blocks(self.__sweetness_limit, dtype=X.dtype)
        env = super()._lazy_transform_gpu(X)

        return env / inst_freq

    def _lazy_transform_cpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_cpu(X)
        inst_freq = inst_freq.map_blocks(self.__sweetness_limit, dtype=X.dtype)
        env = super()._lazy_transform_cpu(X)

        return env / inst_freq

    def _transform_gpu(self, X):
        inst_freq = self.__inst_freq._transform_gpu(X)
        inst_freq = self.__sweetness_limit(inst_freq)
        env = super()._transform_gpu(X)

        return env / inst_freq

    def _transform_cpu(self, X):
        inst_freq = self.__inst_freq._transform_cpu(X)
        inst_freq = self.__sweetness_limit(inst_freq)
        env = super()._transform_cpu(X)

        return env / inst_freq


class QualityFactor(InstantaneousFrequency):
    def __init__(self, sample_rate=4):
        super().__init__(sample_rate=sample_rate)

        self.__rac = RelativeAmplitudeChange()

    def _lazy_transform_gpu(self, X):
        inst_freq = super()._lazy_transform_gpu(X)
        rac = self.__rac._lazy_transform_gpu(X)
        result = (np.pi * inst_freq) / rac
        
        result[da.isnan(result)] = 0
        
        return result

    def _lazy_transform_cpu(self, X):
        inst_freq = super()._lazy_transform_cpu(X)
        rac = self.__rac._lazy_transform_cpu(X)
        result = (np.pi * inst_freq) / rac
        
        result[da.isnan(result)] = 0
        
        return result

    def _transform_gpu(self, X):
        inst_freq = super()._transform_gpu(X)
        rac = self.__rac._transform_gpu(X)
        result = (np.pi * inst_freq) / rac
        
        result[da.isnan(result)] = 0
        
        return result

    def _transform_cpu(self, X):
        inst_freq = super()._transform_cpu(X)
        rac = self.__rac._transform_cpu(X)
        result = (np.pi * inst_freq) / rac
        
        result[da.isnan(result)] = 0
        
        return result


class ResponsePhase(Transform):
    def __init__(self):
        super().__init__()

        self.__envelope = Envelope()
        self.__inst_phase = InstantaneousPhase()

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        phase = self.__inst_phase._lazy_transform_gpu(X)

        troughs = env.map_blocks(local_events, comparator=cp.less,
                                 is_cupy=True, dtype=X.dtype,
                                 meta=cp.array((), dtype=X.dtype))

        troughs = troughs.cumsum(axis=-1)
        result = da.map_blocks(response_operation_gpu, env, phase, troughs,
                               dtype=X.dtype, meta=cp.array((), dtype=X.dtype))
        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        phase = self.__inst_phase._lazy_transform_cpu(X)

        troughs = env.map_blocks(local_events, comparator=np.less,
                                 is_cupy=False, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)
        result = da.map_blocks(response_operation_cpu, env, phase, troughs,
                               dtype=X.dtype)
        result[da.isnan(result)] = 0

        return result

    
    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        phase = self.__inst_phase._transform_gpu(X)
        troughs = local_events(env, comparator=cp.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_gpu(env, phase, troughs)

        result[da.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        phase = self.__inst_phase._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_cpu(env, phase, troughs)

        result[da.isnan(result)] = 0

        return result


class ResponseFrequency(Transform):
    def __init__(self, sample_rate=4):
        super().__init__()

        self.__envelope = Envelope()
        self.__inst_freq = InstantaneousFrequency(sample_rate=sample_rate)

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        inst_freq = self.__inst_freq._lazy_transform_gpu(X)
        troughs = env.map_blocks(local_events, comparator=cp.less,
                                 is_cupy=True, dtype=X.dtype,
                                 meta=cp.array((), dtype=X.dtype))

        troughs = troughs.cumsum(axis=-1)
        result = da.map_blocks(response_operation_gpu, env, inst_freq, troughs,
                               dtype=X.dtype, meta=cp.array((), dtype=X.dtype))
        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        inst_freq = self.__inst_freq._lazy_transform_cpu(X)
        troughs = env.map_blocks(local_events, comparator=np.less,
                                 is_cupy=False, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)
        result = da.map_blocks(response_operation_cpu, env, inst_freq, troughs,
                               dtype=X.dtype)
        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        inst_freq = self.__inst_freq._transform_gpu(X)
        troughs = local_events(env, comparator=cp.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_gpu(env, inst_freq, troughs)

        result[da.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        inst_freq = self.__inst_freq._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_cpu(env, inst_freq, troughs)

        result[da.isnan(result)] = 0

        return result


class ResponseAmplitude(Transform):
    def __init__(self):
        super().__init__()

        self.__envelope = Envelope()

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        troughs = env.map_blocks(local_events, comparator=cp.less,
                                 is_cupy=True, dtype=X.dtype,
                                 meta=cp.array((), dtype=X.dtype))

        troughs = troughs.cumsum(axis=-1)

        X = X.rechunk(env.chunks)

        result = da.map_blocks(response_operation_gpu, env, X, troughs,
                               dtype=X.dtype, meta=cp.array((), dtype=X.dtype))

        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        troughs = env.map_blocks(local_events, comparator=np.less,
                                 is_cupy=False, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)

        X = X.rechunk(env.chunks)

        result = da.map_blocks(response_operation_cpu, env, X, troughs,
                               dtype=X.dtype)

        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        troughs = local_events(env, comparator=cp.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_gpu(env, X, troughs)

        result[da.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = response_operation_cpu(env, X, troughs)

        result[da.isnan(result)] = 0

        return result


class ApparentPolarity(Transform):
    def __init__(self):
        super().__init__()

        self.__envelope = Envelope()

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        troughs = env.map_blocks(local_events, comparator=cp.less,
                                 is_cupy=True, dtype=X.dtype,
                                 meta=cp.array((), dtype=X.dtype))

        troughs = troughs.cumsum(axis=-1)

        X = X.rechunk(env.chunks)

        result = da.map_blocks(polarity_operation_gpu, env, X, troughs,
                               dtype=X.dtype, meta=cp.array((), dtype=X.dtype))

        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        troughs = env.map_blocks(local_events, comparator=np.less,
                                 is_cupy=False, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)

        X = X.rechunk(env.chunks)

        result = da.map_blocks(polarity_operation_cpu, env, X, troughs,
                               dtype=X.dtype)

        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        troughs = local_events(env, comparator=cp.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = polarity_operation_gpu(env, X, troughs)

        result[da.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = polarity_operation_cpu(env, X, troughs)

        result[da.isnan(result)] = 0

        return result

# Numba functions
@njit(parallel=False)
def response_operation_cpu(chunk1, chunk2, chunk3):
    out = np.zeros_like(chunk1)
    for i, j in np.ndindex(out.shape[:-1]):
        ints = np.unique(chunk3[i, j, :])
        for ii in ints:
            idx = np.where(chunk3[i, j, :] == ii)
            idx = idx[0]
            ind = np.zeros(idx.shape[0])
            for k in range(len(idx)):
                ind[k] = chunk1[i, j, idx[k]]
            ind = ind.argmax()
            peak = idx[ind]
            for k in range(len(idx)):
                out[i, j, idx[k]] = chunk2[i, j, peak]
    return out

def response_operation_gpu(chunk1, chunk2, chunk3):
    out = cp.zeros_like(chunk1)
    uniques = cp.zeros_like(chunk1)
    max_ind = cp.zeros_like(chunk1, dtype='int32')
    blockx = out.shape[0]
    blocky = (out.shape[1] + 64 - 1) // 64
    kernel = response_kernel_gpu[(blockx, blocky), (1, 64)]
    kernel(chunk1, chunk2, chunk3, out, uniques, max_ind, out.shape[0], out.shape[1], out.shape[2])
    return out


@cuda.jit()
def response_kernel_gpu(chunk1, chunk2, chunk3, out, uniques, max_ind, len_x, len_y, len_z):
    i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    if i < len_x and j < len_y:
        tail_unique = 0
        for k in range(len_z):
            exists = 0
            val = chunk3[i, j, k]
            for l in range(len_z):
                if l < tail_unique:
                    if val == uniques[i, j, l]:
                        exists = 1
                        if chunk1[i, j, k] > chunk1[i, j, max_ind[i, j, l]]:
                            max_ind[i, j, l] = k
            if exists == 0:
                uniques[i, j, tail_unique] = val
                max_ind[i, j, tail_unique] = k
                tail_unique += 1
        for k in range(len_z):
            if k < tail_unique:
                peak_val = chunk2[i, j, max_ind[i, j, k]]
                for l in range(len_z):
                    if chunk3[i, j, l] == uniques[i, j, k]:
                        out[i, j, l] = peak_val


@njit(parallel=False)
def polarity_operation_cpu(chunk1, chunk2, chunk3):
    out = np.zeros_like(chunk1)
    for i, j in np.ndindex(out.shape[:-1]):
        ints = np.unique(chunk3[i, j, :])
        for ii in ints:
            idx = np.where(chunk3[i, j, :] == ii)
            idx = idx[0]
            ind = np.zeros(idx.shape[0])
            for k in range(len(idx)):
                ind[k] = chunk1[i, j, idx[k]]
            ind = ind.argmax()
            peak = idx[ind]
            val = chunk1[i, j, peak] * np.sign(chunk2[i, j, peak])
            for k in range(len(idx)):
                out[i, j, idx[k]] = val
    return out


def polarity_operation_gpu(chunk1, chunk2, chunk3):
    out = cp.zeros_like(chunk1)
    uniques = cp.zeros_like(chunk1)
    max_ind = cp.zeros_like(chunk1, dtype='int32')
    blockx = out.shape[0]
    blocky = (out.shape[1] + 64 - 1) // 64
    kernel = polarity_kernel_gpu[(blockx, blocky), (1, 64)]
    kernel(chunk1, chunk2, chunk3, out, uniques, max_ind, out.shape[0], out.shape[1], out.shape[2])
    return out


@cuda.jit()
def polarity_kernel_gpu(chunk1, chunk2, chunk3, out, uniques, max_ind, len_x, len_y, len_z):
    i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    j = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    if i < len_x and j < len_y:
        tail_unique = 0
        for k in range(len_z):
            exists = 0
            val = chunk3[i, j, k]
            for l in range(len_z):
                if l < tail_unique:
                    if val == uniques[i, j, l]:
                        exists = 1
                        if chunk1[i, j, k] > chunk1[i, j, max_ind[i, j, l]]:
                            max_ind[i, j, l] = k
            if exists == 0:
                uniques[i, j, tail_unique] = val
                max_ind[i, j, tail_unique] = k
                tail_unique += 1
        for k in range(len_z):
            if k < tail_unique:
                peak_val = chunk1[i, j, max_ind[i, j, k]]
                if chunk2[i, j, max_ind[i, j, k]] < 0:
                    peak_val = -peak_val
                elif chunk2[i, j, max_ind[i, j, k]] == 0:
                    peak_val = 0
                for l in range(len_z):
                    if chunk3[i, j, l] == uniques[i, j, k]:
                        out[i, j, l] = peak_val
