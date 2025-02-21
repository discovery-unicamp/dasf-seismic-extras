#!/usr/bin/env python3

import dask.array as da
import numpy as np
from scipy import ndimage as ndi
from scipy import signal

try:
    import cupy as cp
    if cp.cuda.runtime.runtimeGetVersion() >= 12000:
        import cupyx.scipy.signal as cusignal
    else:
        import cusignal

    from cupyx.scipy import ndimage as cundi
except ImportError:
    pass

from dasf.transforms import Transform
from dasf.utils.types import is_gpu_array
from dasf_seismic.utils.utils import dask_overlap, dask_trim_internal

from dasf_seismic_extras.attributes.complex_trace import Hilbert
from dasf_seismic_extras.attributes.dip_azm import GradientStructureTensor
from dasf_seismic_extras.attributes.signal import FirstDerivative
from dasf_seismic_extras.utils.utils import (
    extract_patches,
    inf_to_max_value,
    inf_to_min_value,
)


class Semblance(Transform):
    def __init__(self, kernel=(3, 3, 9)):
        super().__init__()

        self._kernel = kernel

    def _semblance_cu(self, chunk, kernel):
        sembl_code = ("extern \"C\" __global__" +
        f"    void semblance_kernel(const double* x, int filter_size, double* y) {{"
        f"        double s1[{kernel[2]}];" +
        f"        double s2[{kernel[2]}];" +
        f"        double ss = 0;" +
        f"        double ss1 = 0;" +
        f"        double ss2 = 0;" +
        f"        int nsamples = {kernel[0] * kernel[1]};" +
        f"        int ntraces = {kernel[2]};" +
        f"        int k = 0;" +
        f"        " +
        f"        for(k=0; k < ntraces; k++) {{" +
        f"            s1[k] = 0;" +
        f"            s2[k] = 0;" +
        f"        }}" +
        f"        " +
        f"        for (int i=0; i < filter_size; i++) {{" +
        f"            if (i % ntraces == 0) {{" +
        f"                    k = 0;" +
        f"            }} else {{" +
        f"                    k++;" +
        f"            }}" +
        f"            s1[k] = s1[k] + x[i];" +
        f"            s2[k] = s2[k] + (x[i] * x[i]);" +
        f"        }}" +
        f"        " +
        f"        for(k=0; k < ntraces; k++) {{" +
        f"            ss1 = ss1 + s1[k] * s1[k];" +
        f"            ss2 = ss2 + s2[k];" +
        f"        }}" +
        f"        ss = ss1 / ss2;" +
        f"        " +
        f"        y[0] = (double) (ss / nsamples);" +
        f"}}")

        sembl_raw = cp.RawKernel(sembl_code, "semblance_kernel")

        return cundi.generic_filter(input=chunk, function=sembl_raw, size=kernel)

    def _semblance(self, chunk, kernel):
        def operator(x):
            """Function to extract patches and perform algorithm"""

            # We need this workaround because scipy for GPU does not support
            # extra arguments
            if is_gpu_array(chunk):
                xp = cp
            else:
                xp = np

            if hasattr(xp, 'seterr'):
                xp.seterr(all='ignore')

            region = x.reshape(-1, x.shape[-1])
            ntraces, nsamples = region.shape

            s1 = xp.sum(region, axis=0) ** 2
            s2 = xp.sum(region ** 2, axis=0)

            sembl = s1.sum(axis=-1) / s2.sum(axis=-1)
            sembl /= ntraces

            return sembl

        wrapper = lambda x: operator(x.reshape(kernel))

        return ndi.generic_filter(input=chunk, function=wrapper, size=kernel)

    def _lazy_transform(self, X, function, xp):
        # Generate Dask Array as necessary and perform algorithm
        X_da = dask_overlap(X, self._kernel)

        result = X_da.map_blocks(function, kernel=self._kernel,
                                 dtype=X_da.dtype,
                                 meta=xp.array((), dtype=X_da.dtype))

        result = dask_trim_internal(result, self._kernel)

        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, function, xp):
        result = function(X, kernel=self._kernel)

        result[xp.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, self._semblance_cu, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, self._semblance, np)

    def _transform_gpu(self, X):
        return self._transform(X, self._semblance_cu, cp)

    def _transform_cpu(self, X):
        return self._transform(X, self._semblance, np)


class Semblance2(Transform):
    def __init__(self, kernel=(3, 3, 9)):
        super().__init__()

        self._kernel = kernel

    def __operation(self, chunk, kernel, xp):
        """Function to extract patches and perform algorithm"""
        if hasattr(xp, 'seterr'):
            xp.seterr(all='ignore')

        x = extract_patches(chunk, kernel, xp)

        s1 = xp.sum(x, axis=(-3, -2)) ** 2
        s2 = xp.sum(x ** 2, axis=(-3, -2))

        sembl = s1.sum(axis=-1) / s2.sum(axis=-1)
        sembl /= kernel[0] * kernel[1]

        return sembl

    def _lazy_transform(self, X, xp):
        # Generate Dask Array as necessary and perform algorithm
        X_da = dask_overlap(X, self._kernel)

        result = X_da.map_blocks(self.__operation, kernel=self._kernel,
                                 xp=xp, dtype=X_da.dtype,
                                 meta=xp.array((), dtype=X_da.dtype))

        result = dask_trim_internal(result, self._kernel)

        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, xp):
        result = self.__operation(X, kernel=self._kernel, xp=xp)

        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, np)

    def _transform_gpu(self, X):
        return self._transform(X, cp)

    def _transform_cpu(self, X):
        return self._transform(X, np)


class EigComplex(Hilbert):
    def __init__(self, kernel=(3, 3, 9)):
        super().__init__()

        self._kernel = kernel

    def __cov(self, x, ki, kj, kk, xp):
        """Function to compute the COV"""
        x = x.reshape((ki * kj, kk))
        x = xp.hstack([x.real, x.imag])
        res = x.dot(x.T)

        res[res == xp.inf] = inf_to_max_value(res, xp)
        res[res == -xp.inf] = inf_to_min_value(res, xp)

        res[xp.isnan(res)] = 0

        return res

    def __operation(self, chunk, kernel, xp):
        """Function to extract patches and perform algorithm"""
        ki, kj, kk = kernel
        patches = extract_patches(chunk, kernel, xp)

        if not hasattr(xp.linalg, 'eigvals'):
            raise NotImplementedError("eigvals() function is not "
                                      "available in cp.linalg.")

        out_data = []
        for i in range(0, patches.shape[0]):
            traces = patches[i]
            traces = traces.reshape(-1, ki * kj * kk)
            cova = xp.apply_along_axis(self.__cov, 1, traces, ki, kj, kk, xp)
            vals = xp.linalg.eigvals(cova)
            vals = xp.abs(vals.max(axis=1) / vals.sum(axis=1))

            out_data.append(vals)

        return xp.asarray(out_data).reshape(patches.shape[:3])

    def _lazy_transform(self, X, xsignal, xp):
        hilbert = super()._lazy_transform(X, xsignal, xp)

        X_da = dask_overlap(hilbert, self._kernel)

        result = X_da.map_blocks(self.__operation, kernel=self._kernel,
                                    xp=xp, dtype=X_da.dtype)
        
        result = dask_trim_internal(result, self._kernel)

        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, signal, np)

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cusignal, cp)

    def _transform_cpu(self, X):
        hilbert = super()._transform_cpu(X)

        result = self.__operation(hilbert, kernel=self._kernel, xp=np)

        result[np.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        hilbert = super()._transform_gpu(X)

        result = self.__operation(hilbert, kernel=self._kernel, xp=cp)

        result[cp.isnan(result)] = 0

        return result


class Chaos(Transform):
    def __init__(self, kernel=(3, 3, 9)):
        super().__init__()

        self._kernel = kernel

        self.__gst = GradientStructureTensor(kernel=self._kernel)

    def __operation(self, gi2, gj2, gk2, gigj, gigk, gjgk, xp):
        if hasattr(xp, 'seterr'):
            xp.seterr(all='ignore')

        chunk_shape = gi2.shape

        gst = xp.array([[gi2, gigj, gigk],
                        [gigj, gj2, gjgk],
                        [gigk, gjgk, gk2]])

        gst = xp.moveaxis(gst, [0, 1], [-2, -1])
        gst = gst.reshape((-1, 3, 3))

        eigs = xp.sort(xp.linalg.eigvalsh(gst))

        e1 = eigs[:, 2].reshape(chunk_shape)
        e2 = eigs[:, 1].reshape(chunk_shape)
        e3 = eigs[:, 0].reshape(chunk_shape)

        out = (2 * e2) / (e1 + e3)

        return out

    def _lazy_transform_gpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_gpu(X)

        return da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, xp=cp, dtype=gi2.dtype,
                               meta=cp.array((), dtype=gi2.dtype))

    def _lazy_transform_cpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_cpu(X)

        return da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, xp=np, dtype=gi2.dtype,
                               meta=np.array((), dtype=gi2.dtype))

    def _transform_gpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_gpu(X)

        return self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, cp)

    def _transform_cpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_cpu(X)

        return self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, np)


class Coherence(Transform):
    def __init__(self, kernel=(3, 3, 9)):
        super().__init__()

        self._kernel = kernel

        self.__gst = GradientStructureTensor(kernel=self._kernel)

    def __operation(self, gi2, gj2, gk2, gigj, gigk, gjgk, xp):
        if hasattr(xp, 'seterr'):
            xp.seterr(all='ignore')

        chunk_shape = gi2.shape

        gst = xp.array([[gi2, gigj, gigk],
                        [gigj, gj2, gjgk],
                        [gigk, gjgk, gk2]])

        gst = xp.moveaxis(gst, [0, 1], [-2, -1])
        gst = gst.reshape((-1, 3, 3))

        eigs = xp.sort(xp.linalg.eigvalsh(gst))

        e1 = eigs[:, 2].reshape(chunk_shape)
        e2 = eigs[:, 1].reshape(chunk_shape)

        out = (e1 - e2) / (e1 + e2)

        return out

    def _lazy_transform_gpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_gpu(X)

        return da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, xp=cp, dtype=gi2.dtype,
                               meta=cp.array((), dtype=gi2.dtype))

    def _lazy_transform_cpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_cpu(X)

        return da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, xp=np, dtype=gi2.dtype,
                               meta=np.array((), dtype=gi2.dtype))

    def _transform_gpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_gpu(X)

        return self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, cp)

    def _transform_cpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_cpu(X)

        return self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, np)


class StructuredSemblance(Transform):
    def __init__(self, kernel=(3, 3, 9)):
        super().__init__()

        self._kernel = kernel

        self.__gst = GradientStructureTensor(kernel=self._kernel)

    def __operation(self, gi2, gj2, gk2, gigj, gigk, gjgk, xp):
        if hasattr(xp, 'seterr'):
            xp.seterr(all='ignore')

        chunk_shape = gi2.shape

        gst = xp.array([[gi2, gigj, gigk],
                        [gigj, gj2, gjgk],
                        [gigk, gjgk, gk2]])

        gst = xp.moveaxis(gst, [0, 1], [-2, -1])
        gst = gst.reshape((-1, 3, 3))

        eigs = xp.sort(xp.linalg.eigvalsh(gst))

        e1 = eigs[:, 2].reshape(chunk_shape)
        e2 = eigs[:, 1].reshape(chunk_shape)
        e3 = eigs[:, 0].reshape(chunk_shape)

        out = (((e2 + e3) * (e2 + e3)) * e1) / (((e2 * e2) + (e3 * e3)) * e1)

        return out

    def _lazy_transform_gpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_gpu(X)

        return da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, xp=cp, dtype=gi2.dtype,
                               meta=cp.array((), dtype=gi2.dtype))

    def _lazy_transform_cpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_cpu(X)

        return da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, xp=np, dtype=gi2.dtype,
                               meta=np.array((), dtype=gi2.dtype))

    def _transform_gpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_gpu(X)

        return self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, cp)

    def _transform_cpu(self, X):
        # Compute GST
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_cpu(X)

        return self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, np)


class VolumeCurvature(Transform):
    def __init__(self, dip_factor=10, kernel=(3, 3, 3)):
        super().__init__()

        self._dip_factor = dip_factor
        self._kernel = kernel

        self.__first_derivative_ux = FirstDerivative(axis=0)
        self.__first_derivative_uy = FirstDerivative(axis=1)
        self.__first_derivative_uz = FirstDerivative(axis=2)
        self.__first_derivative_vx = FirstDerivative(axis=0)
        self.__first_derivative_vy = FirstDerivative(axis=1)
        self.__first_derivative_vz = FirstDerivative(axis=2)

    def _lazy_transform(self, u, v, ux, uy, uz, vx, vy, vz, xndi):
        hw = tuple(np.array(self._kernel) // 2)
        ux = ux.map_overlap(xndi.uniform_filter, size=self._kernel,
                           depth=hw, boundary="reflect", dtype=ux.dtype)
        uy = uy.map_overlap(xndi.uniform_filter, size=self._kernel,
                           depth=hw, boundary="reflect", dtype=ux.dtype)
        uz = uz.map_overlap(xndi.uniform_filter, size=self._kernel,
                           depth=hw, boundary="reflect", dtype=ux.dtype)
        vx = vx.map_overlap(xndi.uniform_filter, size=self._kernel,
                           depth=hw, boundary="reflect", dtype=ux.dtype)
        vy = vy.map_overlap(xndi.uniform_filter, size=self._kernel,
                           depth=hw, boundary="reflect", dtype=ux.dtype)
        vz = vz.map_overlap(xndi.uniform_filter, size=self._kernel,
                           depth=hw, boundary="reflect", dtype=ux.dtype)

        w = da.ones_like(u)
        wx = da.zeros_like(ux)
        wy = da.zeros_like(ux)
        wz = da.zeros_like(ux)

        uv = u * v
        vw = v * w
        u2 = u * u
        v2 = v * v
        w2 = w * w
        u2pv2 = u2 + v2
        v2pw2 = v2 + w2
        s = da.sqrt(u2pv2 + w2)

        # Measures of surfaces
        E = da.ones_like(u)
        F = -(u * w) / (da.sqrt(u2pv2) * da.sqrt(v2pw2))
        G = da.ones_like(u)
        D = -(-uv * vx+u2 * vy + v2 * ux - uv * uy) / (u2pv2 * s)
        Di = -(vw * (uy + vx) - 2 * u * w * vy - v2 *
               (uz + wx) + uv * (vz + wy)) / (2 * da.sqrt(u2pv2) *
                                              da.sqrt(v2pw2) * s)
        Dii = -(-vw * wy + v2 * wz + w2 * vy - vw * vz) / (v2pw2 * s)
        H = (E * Dii - 2 * F * Di + G * D) / (2 * (E * G - F * F))
        K = (D * Dii - Di * Di) / (E * G - F * F)
        Kmin = H - da.sqrt(H * H - K)
        Kmax = H + da.sqrt(H * H - K)

        H[da.isnan(H)] = 0
        K[da.isnan(K)] = 0
        Kmax[da.isnan(Kmax)] = 0
        Kmin[da.isnan(Kmin)] = 0

        KMPos = da.maximum(Kmax, Kmin)
        KMNeg = da.minimum(Kmax, Kmin)

        return (H, K, Kmax, Kmin, KMPos, KMNeg)

    def _transform(self, u, v, ux, uy, uz, vx, vy, vz, xndi, xp):
        w = xp.ones_like(u)

        ux = xndi.uniform_filter(ux, size=self._kernel)
        uy = xndi.uniform_filter(uy, size=self._kernel)
        uz = xndi.uniform_filter(uz, size=self._kernel)
        vx = xndi.uniform_filter(vx, size=self._kernel)
        vy = xndi.uniform_filter(vy, size=self._kernel)
        vz = xndi.uniform_filter(vz, size=self._kernel)

        wx = xp.zeros_like(ux)
        wy = xp.zeros_like(ux)
        wz = xp.zeros_like(ux)

        uv = u * v
        vw = v * w
        u2 = u * u
        v2 = v * v
        w2 = w * w
        u2pv2 = u2 + v2
        v2pw2 = v2 + w2
        s = xp.sqrt(u2pv2 + w2)

        # Measures of surfaces
        E = xp.ones_like(u)
        F = -(u * w) / (xp.sqrt(u2pv2) * xp.sqrt(v2pw2))
        G = xp.ones_like(u)
        D = -(-uv * vx+u2 * vy + v2 * ux - uv * uy) / (u2pv2 * s)
        Di = -(vw * (uy + vx) - 2 * u * w * vy - v2 *
               (uz + wx) + uv * (vz + wy)) / (2 * xp.sqrt(u2pv2) *
                                              xp.sqrt(v2pw2) * s)
        Dii = -(-vw * wy + v2 * wz + w2 * vy - vw * vz) / (v2pw2 * s)
        H = (E * Dii - 2 * F * Di + G * D) / (2 * (E * G - F * F))
        K = (D * Dii - Di * Di) / (E * G - F * F)
        Kmin = H - xp.sqrt(H * H - K)
        Kmax = H + xp.sqrt(H * H - K)

        H[xp.isnan(H)] = 0
        K[xp.isnan(K)] = 0
        Kmax[xp.isnan(Kmax)] = 0
        Kmin[xp.isnan(Kmin)] = 0

        KMPos = xp.maximum(Kmax, Kmin)
        KMNeg = xp.minimum(Kmax, Kmin)

        return (H, K, Kmax, Kmin, KMPos, KMNeg)

    def _lazy_transform_gpu(self, X_il, X_xl):
        u = -X_il / self._dip_factor
        v = -X_xl / self._dip_factor

        # Compute Gradients
        ux = self.__first_derivative_ux._lazy_transform_gpu(u)
        uy = self.__first_derivative_uy._lazy_transform_gpu(u)
        uz = self.__first_derivative_uz._lazy_transform_gpu(u)
        vx = self.__first_derivative_vx._lazy_transform_gpu(v)
        vy = self.__first_derivative_vy._lazy_transform_gpu(v)
        vz = self.__first_derivative_vz._lazy_transform_gpu(v)

        return self._lazy_transform(u, v, ux, uy, uz, vx, vy, vz, cundi)

    def _lazy_transform_cpu(self, X_il, X_xl):
        u = -X_il / self._dip_factor
        v = -X_xl / self._dip_factor

        # Compute Gradients
        ux = self.__first_derivative_ux._lazy_transform_cpu(u)
        uy = self.__first_derivative_uy._lazy_transform_cpu(u)
        uz = self.__first_derivative_uz._lazy_transform_cpu(u)
        vx = self.__first_derivative_vx._lazy_transform_cpu(v)
        vy = self.__first_derivative_vy._lazy_transform_cpu(v)
        vz = self.__first_derivative_vz._lazy_transform_cpu(v)

        return self._lazy_transform(u, v, ux, uy, uz, vx, vy, vz, ndi)

    def _transform_gpu(self, X_il, X_xl):
        u = -X_il / self._dip_factor
        v = -X_xl / self._dip_factor

        # Compute Gradients
        ux = self.__first_derivative_ux._transform_gpu(u)
        uy = self.__first_derivative_uy._transform_gpu(u)
        uz = self.__first_derivative_uz._transform_gpu(u)
        vx = self.__first_derivative_vx._transform_gpu(v)
        vy = self.__first_derivative_vy._transform_gpu(v)
        vz = self.__first_derivative_vz._transform_gpu(v)

        return self._transform(u, v, ux, uy, uz, vx, vy, vz, cundi, cp)

    def _transform_cpu(self, X_il, X_xl):
        u = -X_il / self._dip_factor
        v = -X_xl / self._dip_factor

        # Compute Gradients
        ux = self.__first_derivative_ux._transform_cpu(u)
        uy = self.__first_derivative_uy._transform_cpu(u)
        uz = self.__first_derivative_uz._transform_cpu(u)
        vx = self.__first_derivative_vx._transform_cpu(v)
        vy = self.__first_derivative_vy._transform_cpu(v)
        vz = self.__first_derivative_vz._transform_cpu(v)

        return self._transform(u, v, ux, uy, uz, vx, vy, vz, ndi, np)
