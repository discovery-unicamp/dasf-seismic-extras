#!/usr/bin/env python3

import dask.array as da
import numpy as np
import scipy.fft as fft_sp
from numpy.fft import fft as fft_np
from numpy.fft import ifft as ifft_np
from scipy import signal
from scipy.fft import fftfreq

try:
    import cupy as cp
    if cp.cuda.runtime.runtimeGetVersion() >= 12000:
        import cupyx.scipy.signal as cusignal
    else:
        import cusignal

    from cupy.fft import fft as fft_cp
    from cupy.fft import fftfreq as fftfreq_cp
    from cupy.fft import ifft as ifft_cp
except ImportError:
    pass

from dasf.transforms import Transform
from dasf_seismic.utils.utils import dask_overlap

from dasf_seismic_extras.utils.utils import matching_dtypes


class LowpassFilter(Transform):
    def __init__(self, freq, sample_rate=4):
        super().__init__()

        self._freq = freq
        self._sample_rate = sample_rate

        self.__btype = 'lowpass'

    def __operation(self, chunk, B, A, xsignal):
        """Filtering Function"""
        return xsignal.filtfilt(B, A, x=chunk)

    def _lazy_transform(self, X, xsignal, xp):
        # Generate Dask Array as necessary and perform algorithm
        X_da = dask_overlap(X, kernel=None)

        fs = 1000 / self._sample_rate
        nyq = fs * 0.5
        B, A = xsignal.butter(6, self._freq/nyq, btype=self.__btype,
                             analog=False)

        A = xp.asarray(A, dtype=X.dtype)
        B = xp.asarray(B, dtype=X.dtype)

        return X_da.map_blocks(self.__operation, B, A, xsignal,
                               dtype=X_da.dtype,
                               meta=xp.array((), dtype=X_da.dtype))

    def _transform(self, X, xsignal, xp):
        fs = 1000 / self._sample_rate
        nyq = fs * 0.5
        B, A = xsignal.butter(6, self._freq/nyq, btype=self.__btype,
                             analog=False)

        A = xp.asarray(A, dtype=X.dtype)
        B = xp.asarray(B, dtype=X.dtype)

        return self.__operation(X, B, A, xsignal)

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cusignal, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, signal, np)

    def _transform_gpu(self, X):
        return self._transform(X, cusignal, cp)

    def _transform_cpu(self, X):
        return self._transform(X, signal, np)


class HighpassFilter(LowpassFilter):
    def __init__(self, freq, sample_rate=4):
        super().__init__(freq=freq, sample_rate=sample_rate)

        self.__btype = 'highpass'


class BandpassFilter(Transform):
    def __init__(self, freq_lp, freq_hp, sample_rate=4):
        super().__init__()

        self._freq_lp = freq_lp
        self._freq_hp = freq_hp
        self._sample_rate = sample_rate

        self.__btype = 'bandpass'

    def __operation(self, chunk, B, A, xsignal):
        """Filtering Function"""
        return xsignal.filtfilt(B, A, x=chunk)

    def _lazy_transform(self, X, xsignal, xp):
        # Generate Dask Array as necessary and perform algorithm
        X_da = dask_overlap(X, kernel=None)

        fs = 1000 / self._sample_rate
        nyq = fs * 0.5
        B, A = xsignal.butter(6, (self._freq_lp/nyq, self._freq_hp/nyq),
                             btype=self.__btype, analog=False)

        A = xp.asarray(A, dtype=X.dtype)
        B = xp.asarray(B, dtype=X.dtype)

        return X_da.map_blocks(self.__operation, B, A, xsignal,
                               dtype=X_da.dtype,
                               meta=xp.array((), dtype=X_da.dtype))

    def _transform(self, X, xsignal, xp):
        fs = 1000 / self._sample_rate
        nyq = fs * 0.5
        B, A = xsignal.butter(6, (self._freq_lp/nyq, self._freq_hp/nyq),
                             btype=self.__btype, analog=False)

        A = xp.asarray(A, dtype=X.dtype)
        B = xp.asarray(B, dtype=X.dtype)

        return self.__operation(X, B, A, xsignal)

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cusignal, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, signal, np)

    def _transform_gpu(self, X):
        return self._transform(X, cusignal, cp)

    def _transform_cpu(self, X):
        return self._transform(X, signal, np)


class CWTRicker(Transform):
    def __init__(self, freq, sample_rate=4):
        super().__init__()

        self._freq = freq
        self._sample_rate = sample_rate

    def __wavelet(self, xp):
        """Generate wavelet of specified frequency"""
        sr = self._sample_rate / 1000
        t = xp.arange(-0.512 / 2, 0.512 / 2, sr)
        return ((1 - (2 * (xp.pi * self._freq * t) ** 2)) *
                xp.exp(-(xp.pi * self._freq * t) ** 2))

    def __convolve(self, chunk, w, xsignal, xp):
        """Convolve wavelet with trace"""
        out = xp.zeros_like(chunk)

        for i, j in np.ndindex(chunk.shape[:-1]):
            out[i, j] = xsignal.fftconvolve(chunk[i, j], w,
                                            mode='same')

        return out

    def _lazy_transform(self, X, xsignal, xp):
        # Generate Dask Array as necessary and perform algorithm
        X_da = dask_overlap(X, kernel=None)

        w = self.__wavelet(xp)

        return X_da.map_blocks(self.__convolve, w=w, xsignal=xsignal,
                               xp=xp, dtype=X.dtype)

    def _transform(self, X, xsignal, xp):
        w = self.__wavelet(xp)

        return self.__convolve(X, w=w, xsignal=xsignal, xp=xp)

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cusignal, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, signal, np)

    def _transform_gpu(self, X):
        return self._transform(X, cusignal, cp)

    def _transform_cpu(self, X):
        return self._transform(X, signal, np)


class CWTOrmsby(Transform):
    def __init__(self, freqs, sample_rate=4):
        super().__init__()

        self._freqs = freqs
        self._sample_rate = sample_rate

    def __wavelet(self, xp):
        """Generate wavelet of specified frequencyies"""
        f1, f2, f3, f4 = self._freqs
        sr = self._sample_rate / 1000

        t = xp.arange(-0.512 / 2, 0.512 / 2, sr)

        term1 = ((((xp.pi * f4) ** 2) / ((xp.pi * f4) - (xp.pi * f3))) *
                 xp.sinc(xp.pi * f4 * t) ** 2)
        term2 = ((((xp.pi * f3) ** 2) / ((xp.pi * f4) - (xp.pi * f3))) *
                 xp.sinc(xp.pi * f3 * t) ** 2)
        term3 = ((((xp.pi * f2) ** 2) / ((xp.pi * f2) - (xp.pi * f1))) *
                 xp.sinc(xp.pi * f2 * t) ** 2)
        term4 = ((((xp.pi * f1) ** 2) / ((xp.pi * f2) - (xp.pi * f1))) *
                 xp.sinc(xp.pi * f1 * t) ** 2)

        return (term1 - term2) - (term3 - term4)

    def __convolve(self, chunk, w, xsignal, xp):
        """Convolve wavelet with trace"""
        out = xp.zeros_like(chunk)

        for i, j in np.ndindex(chunk.shape[:-1]):
            out[i, j] = xsignal.fftconvolve(chunk[i, j], w,
                                            mode='same')

        return out

    def _lazy_transform(self, X, xsignal, xp):
        # Generate Dask Array as necessary and perform algorithm
        X_da = dask_overlap(X, kernel=None)

        w = self.__wavelet(xp)

        return X_da.map_blocks(self.__convolve, w=w, xsignal=xsignal,
                               xp=xp, dtype=X.dtype)

    def _transform(self, X, xsignal, xp):
        w = self.__wavelet(xp)

        return self.__convolve(X, w=w, xsignal=xsignal, xp=xp)

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cusignal, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, signal, np)

    def _transform_gpu(self, X):
        return self._transform(X, cusignal, cp)

    def _transform_cpu(self, X):
        return self._transform(X, signal, np)
