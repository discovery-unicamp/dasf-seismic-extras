#!/usr/bin/env python3

import os

import numpy as np
import segyio

try:
    import cupy as cp
except ImportError:
    pass

from segysak import segy


def extract_patches(in_data, kernel, xp):
    """
    Description
    -----------
    Reshape in_data into a collection of patches defined by kernel

    Parameters
    ----------
    in_data : Dask Array, data to convert
    kernel : tuple (len 3), operator size

    Returns
    -------
    out : Numpy Array, has shape (in_data.shape[0], in_data.shape[1],
                                  in_data.shape[2], kernel[0], kernel[1],
                                  kernel[2])
    """

    # This is a workaround for cases where dask chunks are empty
    # Numpy handles if quietly, CuPy does not.
    if in_data.shape == (0, 0, 0):
        return []

    padding = np.array(kernel) // 2
    patches = xp.pad(in_data, ((padding[0], padding[0]),
                               (padding[1], padding[1]),
                               (padding[2], padding[2])),
                     mode='symmetric')
    strides = patches.strides + patches.strides
    shape = tuple(list(patches.shape) + list(kernel))

    patches = xp.lib.stride_tricks.as_strided(patches,
                                              shape=shape,
                                              strides=strides)
    shape = in_data.shape
    patches = patches[:shape[0], :shape[1], :shape[2]]

    return patches


def local_events(in_data, comparator, is_cupy=False):
    """
    Description
    -----------
    Find local peaks or troughs depending on comparator used

    Parameters
    ----------
    in_data : Dask Array, data to convert
    comparator : function, defines truth between neighboring elements

    Keywork Arguments
    -----------------
    is_cupy : handles data directly from GPU

    Returns
    -------
    out : Numpy Array
    """

    if is_cupy:
        idx = cp.arange(0, in_data.shape[-1])
        trace = in_data.take(idx, axis=-1)
        plus = in_data.take(idx + 1, axis=-1)
        minus = in_data.take(idx - 1, axis=-1)
        plus[:, :, -1] = trace[:, :, -1]
        minus[:, :, 0] = trace[:, :, 0]
        result = cp.ones(in_data.shape, dtype=bool)
    else:
        idx = np.arange(0, in_data.shape[-1])
        trace = in_data.take(idx, axis=-1, mode='clip')
        plus = in_data.take(idx + 1, axis=-1, mode='clip')
        minus = in_data.take(idx - 1, axis=-1, mode='clip')

        result = np.ones(in_data.shape, dtype=bool)

    result &= comparator(trace, plus)
    result &= comparator(trace, minus)

    return result


def convert_to_seisnc(segyin, iline=189, xline=193, cdpx=181, cdpy=185):
    """
    Description
    -----------
    Convert SEG-Y file to SEISNC format

    Parameters
    ----------
    segyin : String, path to the SEG-Y file

    Keywork Arguments
    -----------------
    iline : Integer, index of the iline metadata info
    xline : Integer, index of the xline metadata info
    cdpx : Integer, index of the start of the CDPs in X axis
    cdpy : Integer, index of the start of the CDPs in Y axis

    Returns
    -------
    seisnc_path : String, path of the SEISNC file
    """
    directory = os.path.dirname(segyin)

    os.makedirs(directory, exist_ok=True)

    seisnc_file = os.path.splitext(os.path.basename(segyin))[0] + ".seisnc"

    seisnc_path = os.path.join(directory, seisnc_file)

    if os.path.exists(seisnc_path):
        return seisnc_path

    segy.segy_converter(
        segyin, seisnc_path, iline=iline, xline=xline, cdpx=cdpx, cdpy=cdpy
    )

    return seisnc_path


# XXX: Function map_segy needs to open locally the file due to problems of
# serialization when dask transports the segyio object through workers.
def map_segy(x, tmp, contiguous, xp, mode='r', iline=189, xline=193,
             strict=True, ignore_geometry=False, endian='big',
             block_info=None):
    segyfile = segyio.open(tmp, mode=mode, iline=iline, xline=xline,
                           ignore_geometry=ignore_geometry, strict=strict,
                           endian=endian)

    if contiguous:
        loc = block_info[None]['array-location'][0]
        return segyfile.trace.raw[loc[0]:loc[1]]
    else:
        dim_x, dim_y, dim_z = block_info[None]['shape']
        loc_x, loc_y, loc_z = block_info[None]['array-location']
        subcube_x = []
        for i in range(loc_x[0], loc_x[1]):
            subcube_y = []
            for j in range(loc_y[0], loc_y[1]):
                block = segyfile.trace.raw[dim_y * i + j][loc_z[0]:loc_z[1]]
                subcube_y.append(block)
            subcube_x.append(subcube_y)

        return xp.asarray(subcube_x).astype(xp.float64)


def inf_to_max_value(array, xp):
    if array.dtype == xp.float64 or array.dtype == xp.float32:
        return np.finfo(array.dtype).max
    elif array.dtype == xp.int64 or array.dtype == xp.int32:
        return np.iinfo(array.dtype).max


def inf_to_min_value(array, xp):
    if array.dtype == xp.float64 or array.dtype == xp.float32:
        return np.finfo(array.dtype).min
    elif array.dtype == xp.int64 or array.dtype == xp.int32:
        return np.iinfo(array.dtype).min


def set_time_chunk_overlap(dask_array):
    if dask_array.shape[-1] != dask_array.chunksize[-1]:
        print("WARNING: splitting the time axis in chunks can cause "
              "significant performance degradation.")

        time_edge = int(dask_array.chunksize[-1] * 0.1)
        if time_edge < 5:
            time_edge = dask_array.chunksize[-1] * 0.5

        return (1, 1, int(time_edge))
    return None


def dask_cupy_angle_wrapper(data):
    return data.map_blocks(cp.angle, dtype=data.dtype,
                           meta=cp.array((), dtype=data.dtype))


def matching_dtypes(src_dtype, target_dtype, default):
    dtypes = {
        "float32": {
            "int": "int32",
            "complex": "complex64",
        },
        "float64": {
            "int": "int64",
            "complex": "complex128",
        }
    }

    return dtypes.get(str(src_dtype), {}).get(target_dtype, default)
