#!/usr/bin/env python3

import unittest

import dask.array as da
import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

from dasf.transforms import MappedTransform
from dasf.utils.funcs import is_gpu_supported, trim_chunk_location
from dasf_seismic.attributes.texture import LocalBinaryPattern3D

# For LBP Original
from scipy.interpolate import RegularGridInterpolator as RGI
from sklearn import preprocessing


def lbp_original(X):
    """ Scikit's uniform local binary pattern function for 3D matrices."""

    # This code was originally taken from:
    # https://github.com/ajbugge/StratigraphicUnits/blob/master/TextureDescriptor3D.py
    points = []
    for x in range(-1, 2, 2):
        for y in range(-1, 2, 2):
            for z in range(-1, 2, 2):
                if x == 0 and y == 0 and z == 0:
                    continue
                else:
                    point = [x, y, z]
                    points.append(point)
    samples = len(points)
    size_x = X.shape[0]
    size_y = X.shape[1]
    size_z = X.shape[2]

    grid_spacing = (
        np.arange(size_x),
        np.arange(size_y),
        np.arange(size_z)
    )
    interpolator = RGI(grid_spacing, X, bounds_error=False, fill_value=0)
    output = np.zeros(X.shape, dtype=np.uint64)

    weights = 2 ** np.arange(samples, dtype=np.uint64)
    signed_texture = np.zeros(samples, dtype=np.int8)

    for x in range(1, size_x):
        for y in range(1, size_y):
            for z in range(1, size_z):
                center_value = X[x, y, z]
                local_points = np.add(points, [x, y, z])
                sphere_values = interpolator(local_points)
                largest = np.max(sphere_values)
                smallest = np.min(sphere_values)
                sml = True
                lg = True
                for i in range(samples):
                    if sphere_values[i] == largest and sphere_values[i] > center_value and lg is True:
                        signed_texture[i] = 1
                        lg = False
                    elif sphere_values[i] == smallest and sphere_values[i] < center_value and sml is True:
                        signed_texture[i] = 1
                        sml = False
                    else:
                        signed_texture[i] = 0
                for i in range(samples):
                    if signed_texture[i]:
                        output[x, y, z] += weights[i]  # this is the LBP value

    texture_descriptor = output.copy()
    unq = np.unique(output)
    num = 0
    for i in unq:
        texture_descriptor[texture_descriptor == i] = num - 1
        num = num + 1
    return texture_descriptor


def generateFeaturevectorsOrig(seismicCube, textureCube, window_size, step):

    datashape = textureCube.shape
    edge_r = int(window_size[0]/2 - step/2)
    edge_c = int(window_size[1]/2 - step/2)
    edge_d = int(window_size[2]/2 - step/2)
    minseis = np.min(seismicCube)
    maxseis = np.max(seismicCube)
    P = 37

    layercake = np.zeros_like(seismicCube)
    size_z = seismicCube.shape[2]
    for z in range (0, size_z):
        layercake[:, :, z] = z
    
    i = 0

    feature_vectors=[]
    for r in range(0, datashape[0] - window_size[0], step):
        i += 1
        for c in range(0, datashape[1] - window_size[1], step):
            for d in range(0, datashape[2] - window_size[2], step):
                histograms = []

                window = seismicCube[r:r + window_size[0],
                                     c:c + window_size[1],
                                     d:d + window_size[2]]

                hist, _ = np.histogram(window, bins=P, range=(minseis, maxseis))
                histograms.append(hist)

                window = textureCube[r:r + window_size[0],
                                     c:c + window_size[1],
                                     d:d + window_size[2]]
                hist, _ = np.histogram(window, bins=P, range=(0, 36))
                histograms.append(hist)

                feature_vector = np.concatenate(histograms)
                layercakewindow = layercake[r:r + window_size[0],
                                            c:c + window_size[1],
                                            d:d + window_size[2]]

                depth = int(np.mean(layercakewindow))
                feature_vector_depthconstrained = feature_vector + depth
                feature_vector_depthconstrained = feature_vector_depthconstrained.astype(np.float32)
                feature_vectors.append(feature_vector_depthconstrained)

    return np.asarray(feature_vectors)


def generateFeaturevectors(raw_seismic, texture, window_size, step, minseis, maxseis, parent, depth=None, block_info=None):
    if depth is None:
        depth = (0, 0, 0)

    if block_info is not None:
        loc_x, loc_y, loc_z = block_info[0]['array-location']
        block_shape = (loc_x[1] - loc_x[0],
                       loc_y[1] - loc_y[0],
                       loc_z[1] - loc_z[0])

        loc_orig = trim_chunk_location(block_info, depth)

        if ((depth[0] and (loc_x[1] - loc_x[0] - depth[0]) % step) or \
            (depth[1] and (loc_y[1] - loc_y[0] - depth[1]) % step) or \
            (depth[2] and (loc_z[1] - loc_z[0] - depth[2]) % step)):
            raise ValueError("Chunks need to be equally divided by the step.")

        trim = step
    else:
        block_shape = raw_seismic.shape
        loc_x, loc_y, loc_z = ([0, block_shape[0]], [0, block_shape[1]], [0, block_shape[2]])
        loc_orig = ([0, block_shape[0]], [0, block_shape[1]], [0, block_shape[2]])

        trim = 0

    datashape = texture.shape
    P = 37

    # The only difference of the original code is that we use the depth index as Z and not as X.
    # The original code is `layercake[i, :, :] = i`.
    layercake = parent.zeros(block_shape, dtype=parent.uint64)
    for z in range (0, loc_z[1] - loc_z[0]):
        layercake[:, :, z] = z + loc_z[0]

    feature_vectors = []
    for r in range(depth[0], block_shape[0] - (window_size[0] - trim), step):
        for c in range(0, block_shape[1] - window_size[1], step):
            for d in range(0, block_shape[2] - window_size[2], step):
                histograms = []

                window = raw_seismic[r:r + window_size[0],
                                     c:c + window_size[1],
                                     d:d + window_size[2]]

                hist, _ = parent.histogram(window, bins=P, range=(minseis, maxseis))
                histograms.append(hist)

                # Texture is not overlapped. That's why we need the information of the orignal chunk location.
                window = texture[r - depth[0] + loc_orig[0][0]:r - depth[0] + loc_orig[0][0] + window_size[0],
                                 c + loc_y[0]:c + loc_y[0] + window_size[1],
                                 d + loc_z[0]:d + loc_z[0] + window_size[2]]

                hist, _ = parent.histogram(window, bins=P, range=(0, 36))
                histograms.append(hist)

                feature_vector = parent.concatenate(histograms)

                layercakewindow = layercake[r:r + window_size[0],
                                            c:c + window_size[1],
                                            d:d + window_size[2]]

                depth_layercake = parent.int32(parent.nanmean(layercakewindow))
                feature_vector_depthconstrained = feature_vector + depth_layercake
                feature_vector_depthconstrained = feature_vector_depthconstrained.astype(parent.float32)

                feature_vectors.append(feature_vector_depthconstrained)

    return parent.asarray(feature_vectors)


class TestLBPValidation(unittest.TestCase):
    def test_lbp_from_random_numpy_array(self):
        in_shape = (10, 10, 10)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        lbp = LocalBinaryPattern3D()

        lbp_x = lbp._transform_cpu(X=in_data)
        lbp_orig = lbp_original(X=in_data)

        self.assertTrue(np.allclose(lbp_x.astype(np.int8)[1:-1, 1:-1, 1:-1],
                        lbp_orig[1:-1, 1:-1, 1:-1]))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_lbp_from_random_cupy_array(self):
        in_shape = (10, 10, 10)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        lbp = LocalBinaryPattern3D()

        lbp_x = lbp._transform_gpu(X=in_data)
        lbp_orig = lbp_original(X=in_data.get())

        self.assertTrue(np.allclose(lbp_x.get().astype(np.int8)[1:-1, 1:-1, 1:-1],
                        lbp_orig[1:-1, 1:-1, 1:-1]))


class TestGenerateFeatureVectorsValidation(unittest.TestCase):
    def test_gfv_from_random_numpy_array(self):
        in_shape = (40, 40, 40)
        window_size = [5, 5, 5]
        step = 2

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        lbp = LocalBinaryPattern3D()

        lbp_x = lbp._transform_cpu(X=in_data)
        lbp_orig = lbp_original(X=in_data)

        minseis, maxseis = np.min(in_data), np.max(in_data)

        gfv_x = generateFeaturevectors(in_data, lbp_x, window_size, step, minseis, maxseis, parent=np)
        gfv_orig = generateFeaturevectorsOrig(in_data, lbp_x, window_size, step)

        # The original code has an issue with the borders.
        self.assertIsNone(np.testing.assert_array_equal(lbp_x.astype(np.int8)[1:-1, 1:-1, 1:-1],
                                                        lbp_orig[1:-1, 1:-1, 1:-1]))

        self.assertIsNone(np.testing.assert_array_equal(gfv_x, gfv_orig))

        self.assertIsNone(np.testing.assert_array_equal(preprocessing.scale(gfv_x),
                                                        preprocessing.scale(gfv_orig)))

    def test_gfv_from_random_dask_numpy_array_with_step2(self):
        in_shape = (40, 40, 40)
        in_shape_chunks = (10, 40, 40)

        step = 2
        depth = (in_shape_chunks[0] - step, 0, 0)
        window_size = [10, 10, 10]

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        lbp = LocalBinaryPattern3D()

        lbp_x = lbp._transform_cpu(X=in_data)
        lbp_orig = lbp_original(X=in_data)

        minseis, maxseis = np.min(in_data), np.max(in_data)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)
        lbp_x = da.from_array(lbp_x, chunks=in_shape_chunks)

        da_minseis, da_maxseis = da.min(in_data), da.max(in_data)

        gfv = MappedTransform(function=generateFeaturevectors,
                              trim=False,
                              depth=depth,
                              drop_axis=[1, 2],
                              new_axis=[1],
                              boundary='periodic',
                              output_chunk=(1, 74))

        gfv_x = gfv._lazy_transform_cpu(X=in_data, texture=lbp_x, window_size=window_size, step=step, minseis=da_minseis, maxseis=da_maxseis, parent=np, depth=depth)

        gfv_orig = generateFeaturevectorsOrig(in_data, lbp_x, window_size, step)

        self.assertIsNone(np.testing.assert_array_equal(lbp_x.compute().astype(np.int8)[1:-1, 1:-1, 1:-1],
                                                        lbp_orig[1:-1, 1:-1, 1:-1]))

        self.assertIsNone(np.testing.assert_array_equal(gfv_x.compute()[:gfv_orig.shape[0]], gfv_orig))

        self.assertIsNone(np.testing.assert_array_equal(preprocessing.scale(gfv_x.compute()[:gfv_orig.shape[0]]),
                                                        preprocessing.scale(gfv_orig)))

    def test_gfv_from_random_dask_numpy_array_with_step5(self):
        in_shape = (40, 40, 40)
        in_shape_chunks = (10, 40, 40)

        step = 5
        depth = (in_shape_chunks[0] - step, 0, 0)
        window_size = [10, 10, 10]

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        lbp = LocalBinaryPattern3D()

        lbp_x = lbp._transform_cpu(X=in_data)
        lbp_orig = lbp_original(X=in_data)

        minseis, maxseis = np.min(in_data), np.max(in_data)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)
        lbp_x = da.from_array(lbp_x, chunks=in_shape_chunks)

        da_minseis, da_maxseis = da.min(in_data), da.max(in_data)

        gfv = MappedTransform(function=generateFeaturevectors,
                              trim=False,
                              depth=depth,
                              drop_axis=[1, 2],
                              new_axis=[1],
                              boundary='periodic',
                              output_chunk=(1, 74))

        gfv_x = gfv._lazy_transform_cpu(X=in_data, texture=lbp_x, window_size=window_size, step=step, minseis=da_minseis, maxseis=da_maxseis, parent=np, depth=depth)

        gfv_orig = generateFeaturevectorsOrig(in_data, lbp_x, window_size, step)

        self.assertIsNone(np.testing.assert_array_equal(lbp_x.compute().astype(np.int8)[1:-1, 1:-1, 1:-1],
                                                        lbp_orig[1:-1, 1:-1, 1:-1]))

        self.assertIsNone(np.testing.assert_array_equal(gfv_x.compute()[:gfv_orig.shape[0]], gfv_orig))

        self.assertIsNone(np.testing.assert_array_equal(preprocessing.scale(gfv_x.compute()[:gfv_orig.shape[0]]),
                                                        preprocessing.scale(gfv_orig)))
