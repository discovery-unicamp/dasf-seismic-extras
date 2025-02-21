#!/usr/bin/env python3

import dask.array as da
import numpy as np
from scipy import ndimage as ndi

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cundi
except ImportError:
    pass

from dasf.transforms import Transform
from dasf_seismic.utils.utils import dask_overlap

from dasf_seismic_extras.attributes.signal import FirstDerivative


class GradientDips(Transform):
    def __init__(self, dip_factor=10, kernel=(3, 3, 3)):
        super().__init__()

        self._dip_factor = dip_factor
        self._kernel = kernel

        self.__first_derivative_gi = FirstDerivative(axis=0)
        self.__first_derivative_gj = FirstDerivative(axis=1)
        self.__first_derivative_gk = FirstDerivative(axis=2)

    def _lazy_transform(self, il_dip, xl_dip, xndi):
        il_dip[da.isnan(il_dip)] = 0
        xl_dip[da.isnan(xl_dip)] = 0

        # Perform smoothing as specified
        if self._kernel is not None:
            hw = tuple(np.array(self._kernel) // 2)

            il_dip = il_dip.map_overlap(xndi.median_filter, depth=hw,
                                        boundary='reflect',
                                        dtype=il_dip.dtype,
                                        size=self._kernel)
            xl_dip = xl_dip.map_overlap(xndi.median_filter, depth=hw,
                                        boundary='reflect',
                                        dtype=xl_dip.dtype,
                                        size=self._kernel)

        return (il_dip, xl_dip)

    def _transform(self, il_dip, xl_dip, xndi, xp):
        il_dip[xp.isnan(il_dip)] = 0
        xl_dip[xp.isnan(xl_dip)] = 0

        # Perform smoothing as specified
        if self._kernel is not None:
            il_dip = xndi.median_filter(il_dip, size=self._kernel)

            xl_dip = xndi.median_filter(xl_dip, size=self._kernel)

        return (il_dip, xl_dip)

    def _lazy_transform_gpu(self, X):
        X_da = dask_overlap(X, kernel=None)

        # Compute I, J, K gradients
        gi = self.__first_derivative_gi._lazy_transform_gpu(X)
        gj = self.__first_derivative_gj._lazy_transform_gpu(X)
        gk = self.__first_derivative_gk._lazy_transform_gpu(X)

        # Compute dips
        il_dip = -(gi / gk) * self._dip_factor
        xl_dip = -(gj / gk) * self._dip_factor

        return self._lazy_transform(il_dip, xl_dip, cundi)

    def _lazy_transform_cpu(self, X):
        X_da = dask_overlap(X, kernel=None)

        # Compute I, J, K gradients
        gi = self.__first_derivative_gi._lazy_transform_cpu(X)
        gj = self.__first_derivative_gj._lazy_transform_cpu(X)
        gk = self.__first_derivative_gk._lazy_transform_cpu(X)

        # Compute dips
        il_dip = -(gi / gk) * self._dip_factor
        xl_dip = -(gj / gk) * self._dip_factor

        return self._lazy_transform(il_dip, xl_dip, ndi)

    def _transform_gpu(self, X):
        # Compute I, J, K gradients
        gi = self.__first_derivative_gi._transform_gpu(X)
        gj = self.__first_derivative_gj._transform_gpu(X)
        gk = self.__first_derivative_gk._transform_gpu(X)

        # Compute dips
        il_dip = -(gi / gk) * self._dip_factor
        xl_dip = -(gj / gk) * self._dip_factor

        return self._transform(il_dip, xl_dip, cundi, cp)

    def _transform_cpu(self, X):
        # Compute I, J, K gradients
        gi = self.__first_derivative_gi._transform_cpu(X)
        gj = self.__first_derivative_gj._transform_cpu(X)
        gk = self.__first_derivative_gk._transform_cpu(X)

        # Compute dips
        il_dip = -(gi / gk) * self._dip_factor
        xl_dip = -(gj / gk) * self._dip_factor

        return self._transform(il_dip, xl_dip, ndi, np)


class GradientStructureTensor(Transform):
    def __init__(self, kernel):
        super().__init__()

        self._kernel = kernel

        self.__first_derivative_gi = FirstDerivative(axis=0)
        self.__first_derivative_gj = FirstDerivative(axis=1)
        self.__first_derivative_gk = FirstDerivative(axis=2)

    def _lazy_transform(self, gi, gj, gk, xndi):
        """Compute Inner Product of Gradients"""
        hw = tuple(np.array(self._kernel) // 2)

        gi2 = (gi * gi).map_overlap(xndi.uniform_filter, depth=hw,
                                    boundary='reflect',
                                    dtype=gi.dtype, size=self._kernel)
        gj2 = (gj * gj).map_overlap(xndi.uniform_filter, depth=hw,
                                    boundary='reflect',
                                    dtype=gj.dtype, size=self._kernel)
        gk2 = (gk * gk).map_overlap(xndi.uniform_filter, depth=hw,
                                    boundary='reflect',
                                    dtype=gk.dtype, size=self._kernel)
        gigj = (gi * gj).map_overlap(xndi.uniform_filter, depth=hw,
                                     boundary='reflect',
                                     dtype=gj.dtype, size=self._kernel)
        gigk = (gi * gk).map_overlap(xndi.uniform_filter, depth=hw,
                                     boundary='reflect',
                                     dtype=gk.dtype, size=self._kernel)
        gjgk = (gj * gk).map_overlap(xndi.uniform_filter, depth=hw,
                                     boundary='reflect',
                                     dtype=gj.dtype, size=self._kernel)

        return (gi2, gj2, gk2, gigj, gigk, gjgk)

    def _transform(self, gi, gj, gk, xndi):
        gi2 = xndi.uniform_filter(gi * gi, size=self._kernel)
        gj2 = xndi.uniform_filter(gj * gj, size=self._kernel)
        gk2 = xndi.uniform_filter(gk * gk, size=self._kernel)
        gigj = xndi.uniform_filter(gi * gj, size=self._kernel)
        gigk = xndi.uniform_filter(gi * gk, size=self._kernel)
        gjgk = xndi.uniform_filter(gj * gk, size=self._kernel)

        return (gi2, gj2, gk2, gigj, gigk, gjgk)

    def _lazy_transform_gpu(self, X):
        X_da = dask_overlap(X, kernel=None)

        # Compute I, J, K gradients
        gi = self.__first_derivative_gi._lazy_transform_gpu(X)
        gj = self.__first_derivative_gj._lazy_transform_gpu(X)
        gk = self.__first_derivative_gk._lazy_transform_gpu(X)

        return self._lazy_transform(gi, gj, gk, cundi)

    def _lazy_transform_cpu(self, X):
        X_da = dask_overlap(X, kernel=None)

        # Compute I, J, K gradients
        gi = self.__first_derivative_gi._lazy_transform_cpu(X)
        gj = self.__first_derivative_gj._lazy_transform_cpu(X)
        gk = self.__first_derivative_gk._lazy_transform_cpu(X)

        return self._lazy_transform(gi, gj, gk, ndi)

    def _transform_gpu(self, X):
        # Compute I, J, K gradients
        gi = self.__first_derivative_gi._transform_gpu(X)
        gj = self.__first_derivative_gj._transform_gpu(X)
        gk = self.__first_derivative_gk._transform_gpu(X)

        return self._transform(gi, gj, gk, cundi)

    def _transform_cpu(self, X):
        # Compute I, J, K gradients
        gi = self.__first_derivative_gi._transform_cpu(X)
        gj = self.__first_derivative_gj._transform_cpu(X)
        gk = self.__first_derivative_gk._transform_cpu(X)

        return self._transform(gi, gj, gk, ndi)


class GradientStructureTensor2DDips(Transform):
    def __init__(self, dip_factor=10, kernel=(3, 3, 3)):
        super().__init__()

        self._dip_factor = dip_factor
        self._kernel = kernel

        self.__gst = GradientStructureTensor(kernel=kernel)

    def __operation(self, gi2, gj2, gk2, gigj, gigk, gjgk, axis, xp):
        shape = gi2.shape

        gst = xp.array([[gi2, gigj, gigk],
                       [gigj, gj2, gjgk],
                       [gigk, gjgk, gk2]])

        gst = xp.moveaxis(gst, [0, 1], [-2, -1])
        gst = gst.reshape((-1, 3, 3))

        evals, evecs = xp.linalg.eigh(gst)
        ndx = evals.argsort()
        evecs = evecs[xp.arange(0, gst.shape[0], 1), :, ndx[:, 2]]

        out = -evecs[:, axis] / evecs[:, 2]
        out = out.reshape(shape)

        return out

    def _lazy_transform(self, gi2, gj2, gk2, gigj, gigk, gjgk, xp):
        il_dip = da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, axis=0, xp=xp, dtype=gi2.dtype,
                               meta=xp.array((), dtype=gi2.dtype))
        xl_dip = da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, axis=1, xp=xp, dtype=gi2.dtype,
                               meta=xp.array((), dtype=gi2.dtype))

        il_dip *= self._dip_factor
        xl_dip *= self._dip_factor
        il_dip[da.isnan(il_dip)] = 0
        xl_dip[da.isnan(xl_dip)] = 0

        return (il_dip, xl_dip)

    def _transform(self, gi2, gj2, gk2, gigj, gigk, gjgk, xp):
        il_dip = self.__operation(gi2, gj2, gk2, gigj, gigk,
                                  gjgk, axis=0, xp=xp)
        xl_dip = self.__operation(gi2, gj2, gk2, gigj, gigk,
                                  gjgk, axis=1, xp=xp)

        il_dip *= self._dip_factor
        xl_dip *= self._dip_factor
        il_dip[xp.isnan(il_dip)] = 0
        xl_dip[xp.isnan(xl_dip)] = 0

        return (il_dip, xl_dip)

    def _lazy_transform_gpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = \
            self.__gst._lazy_transform_gpu(X)

        return self._lazy_transform(gi2, gj2, gk2, gigj, gigk, gjgk, cp)

    def _lazy_transform_cpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = \
            self.__gst._lazy_transform_cpu(X)

        return self._lazy_transform(gi2, gj2, gk2, gigj, gigk, gjgk, np)

    def _transform_gpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = \
            self.__gst._transform_gpu(X)

        return self._transform(gi2, gj2, gk2, gigj, gigk, gjgk, cp)

    def _transform_cpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = \
            self.__gst._transform_cpu(X)

        return self._transform(gi2, gj2, gk2, gigj, gigk, gjgk, np)


class GradientStructureTensor3DDip(Transform):
    def __init__(self, dip_factor=10, kernel=(3, 3, 3)):
        super().__init__()

        self._dip_factor = dip_factor
        self._kernel = kernel

        self.__gst = GradientStructureTensor(kernel=kernel)

    def __operation(self, gi2, gj2, gk2, gigj, gigk, gjgk, axis, xp):
        """Function to compute 3D dip from GST"""
        if hasattr(xp, 'seterr'):
            xp.seterr(all='ignore')

        shape = gi2.shape

        gst = xp.array([[gi2, gigj, gigk],
                        [gigj, gj2, gjgk],
                        [gigk, gjgk, gk2]])

        gst = xp.moveaxis(gst, [0, 1], [-2, -1])
        gst = gst.reshape((-1, 3, 3))

        evals, evecs = xp.linalg.eigh(gst)
        ndx = evals.argsort()
        evecs = evecs[xp.arange(0, gst.shape[0], 1), :, ndx[:, 2]]

        norm_factor = xp.linalg.norm(evecs, axis=-1)
        evecs[:, 0] /= norm_factor
        evecs[:, 1] /= norm_factor
        evecs[:, 2] /= norm_factor

        evecs *= xp.sign(evecs[:, 2]).reshape(-1, 1)

        dip = xp.dot(evecs, xp.array([0, 0, 1], dtype=evecs.dtype))
        dip = xp.arccos(dip)
        dip = dip.reshape(shape)

        dip = xp.rad2deg(dip)

        return dip

    def _lazy_transform_gpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_gpu(X)

        result = da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, axis=0, xp=cp, dtype=X.dtype,
                               meta=cp.array((), dtype=X.dtype))

        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_cpu(X)

        result = da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, axis=0, xp=np, dtype=X.dtype,
                               meta=np.array((), dtype=X.dtype))

        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_gpu(X)

        result = self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, axis=0,
                                  xp=cp)

        result[cp.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_cpu(X)

        result = self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, axis=0,
                                  xp=np)

        result[np.isnan(result)] = 0

        return result


class GradientStructureTensor3DAzm(Transform):
    def __init__(self, dip_factor=10, kernel=(3, 3, 3)):
        super().__init__()

        self._dip_factor = dip_factor
        self._kernel = kernel

        self.__gst = GradientStructureTensor(kernel=kernel)

    def __operation(self, gi2, gj2, gk2, gigj, gigk, gjgk, axis, xp):
        """Function to compute 3D azimuth from GST"""
        if hasattr(xp, 'seterr'):
            np.seterr(all='ignore')

        shape = gi2.shape

        gst = xp.array([[gi2, gigj, gigk],
                        [gigj, gj2, gjgk],
                        [gigk, gjgk, gk2]])

        gst = xp.moveaxis(gst, [0, 1], [-2, -1])
        gst = gst.reshape((-1, 3, 3))

        evals, evecs = xp.linalg.eigh(gst)
        ndx = evals.argsort()
        evecs = evecs[xp.arange(0, gst.shape[0], 1), :, ndx[:, 2]]

        norm_factor = xp.linalg.norm(evecs, axis=-1)
        evecs[:, 0] /= norm_factor
        evecs[:, 1] /= norm_factor
        evecs[:, 2] /= norm_factor

        evecs *= xp.sign(evecs[:, 2]).reshape(-1, 1)

        azm = xp.arctan2(evecs[:, 0], evecs[:, 1])
        azm = azm.reshape(shape)
        azm = xp.rad2deg(azm)
        azm[azm < 0] += 360

        return azm

    def _lazy_transform_gpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_gpu(X)

        result = da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, axis=0, xp=cp, dtype=X.dtype,
                               meta=cp.array((), dtype=X.dtype))

        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._lazy_transform_cpu(X)

        result = da.map_blocks(self.__operation, gi2, gj2, gk2, gigj, gigk,
                               gjgk, axis=0, xp=np, dtype=X.dtype,
                               meta=np.array((), dtype=X.dtype))

        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_gpu(X)

        result = self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, axis=0,
                                  xp=cp)

        result[cp.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        # Compute Inner Product of Gradients and Dips
        gi2, gj2, gk2, gigj, gigk, gjgk = self.__gst._transform_cpu(X)

        result = self.__operation(gi2, gj2, gk2, gigj, gigk, gjgk, axis=0,
                                  xp=np)

        result[np.isnan(result)] = 0

        return result
