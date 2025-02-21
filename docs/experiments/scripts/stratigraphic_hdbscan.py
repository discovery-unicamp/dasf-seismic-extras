#!/usr/bin/env python3

import sys
import pickle

import cupy as cp
import numpy as np
import dask.array as da

from dasf.datasets import F3
from dasf.transforms import SEGYToArray
from dasf.attributes import LBP3D
from dasf.visualization import Plot2DIline
from dasf.pipeline import ComputePipeline, StagePipeline, MappedStagePipeline
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.preprocessing import StantardScaler
from dasf.ml.cluster import HDBSCAN
from dasf.utils import utils

from pathlib import Path
from datetime import datetime

from dask.distributed import get_task_stream


class Slice(StagePipeline):
    def __init__(self, x=None, y=None, z=None):
        super().__init__(name="Slice data")

        self.block = (x, y, z)

    def run(self, data):

        x, y, z = self.block

        if self.block[0] is None:
            x = data.shape[0]
        if self.block[1] is None:
            y = data.shape[1]
        if self.block[2] is None:
            z = data.shape[2]

        self.block = (x, y, z)

#       return data
        return data[0:self.block[0], 0:self.block[1], 0:self.block[2]]


class MyLBP3D(StagePipeline):
    def __init__(self, **kwargs):
        super().__init__(name="My LBPD3D", **kwargs)

    def run(self, data):
        with open(str(Path.home()) + "/LBP3D_texture.pkl", 'rb') as inp:
            if utils.is_executor_gpu(self.dtype):
                lbp3d = cp.asarray(pickle.load(inp))
            else:
                lbp3d = np.asarray(pickle.load(inp))

        return lbp3d



class TrimSeismicAmplitudes(StagePipeline):
    def __init__(self, percentile, **kwargs):
        super().__init__(name="Trim Seismic Amplitudes", **kwargs)
        
        self.percentile = percentile
    
    def run(self, data):
        start = datetime.now()

        # We need to run it on CPU to reduce GPU memory usage
        data_gpu = data.compute()
        
        if not isinstance(data_gpu, np.ndarray):
            data_cpu = data_gpu.get()
        else:
            data_cpu = data_gpu

        data_p = data_cpu.copy()
        
        data_p[data_p < 0] = 0
        data_p[data_p != 0]

        pos_cutoff = np.percentile(data_p.flatten(), self.percentile) # return percentile
        
        del data_p

        data_n = data_cpu

        data_n = data_n * -1
        data_n[data_n < 0] = 0
        data_n[data_n != 0]
        
        neg_cutoff = -np.percentile(data_n.flatten(), self.percentile) # return percentile
        
        del data_n
        
        neg_cutoff = -9879.0
        pos_cutoff = 8662.0
        
        print ('new min and max are:', neg_cutoff, pos_cutoff)
        data[data > pos_cutoff] = pos_cutoff
        data[data < neg_cutoff] = neg_cutoff

        end = datetime.now()

        print("Time spent to calculate trim:", end - start)

        return data


# Generate feature vectos mapped function
def generate_feature_vectors(block, texture, window_size, step, parent, minseis, maxseis, labels=None, splitlabel=-1, block_info=None):
    if block_info is not None:
        print(block_info[0])
        block_shape = block_info[0]['shape']
        loc_x, loc_y, loc_z = block_info[0]['array-location']
    else:
        block_shape = block.shape
        loc_x, loc_y, loc_z = ([0, block_shape[0]], [0, block_shape[1]], [0, block_shape[2]])
        
    edge_r = parent.int32(window_size[0]/2 - step/2)
    edge_c = parent.int32(window_size[1]/2 - step/2)
    edge_d = parent.int32(window_size[2]/2 - step/2)        
    
    layercake = parent.zeros(block_shape, dtype=parent.uint64)
    for x in range (0, loc_x[1] - loc_x[0]):
        layercake[x, :, :] = x + loc_x[0]
        
    P = 37

    # Just make sure that texture is not a Dask Array
    texture = parent.asarray(texture[loc_x[0]:loc_x[1], loc_y[0]:loc_y[1], loc_z[0]:loc_z[1]])
    
    if isinstance(block, da.core.Array):
        block = block.compute()
        
    if isinstance(minseis, da.core.Array):
        minseis = minseis.compute()
        
    if isinstance(maxseis, da.core.Array):
        maxseis = maxseis.compute()
                        
    feature_vectors = []
    for r in range(0, block_shape[0] - window_size[0], step):
        for c in range(0, block_shape[1] - window_size[1], step):
            for d in range(0, block_shape[2] - window_size[2], step):
                if labels is not None:
                    label_window = labels[r+edge_r:r+step+edge_r,c+edge_c:c+step+edge_c, d+edge_d:d+step+edge_d]

                if splitlabel == -1 or labels is None or label_window[0, 0, 0] == splitlabel:
                    histograms = []
                    
                    window = block[r:r+window_size[0],c:c+window_size[1], d:d+window_size[2]]
                    
                    hist, _ = parent.histogram(window, bins=P, range=(minseis, maxseis))
                    histograms.append(hist)
                    
                    window = texture[r:r+window_size[0],c:c+window_size[1],d:d+window_size[2]]        
                    hist, _ = parent.histogram(window, bins=P, range=(0, 36))
                    histograms.append(hist)

                    feature_vector = parent.concatenate(histograms)
                
                    layercakewindow = layercake[r:r+window_size[0],c:c+window_size[1], d:d+window_size[2]]
                    
                    depth = parent.int(parent.mean(layercakewindow))
                
                    feature_vector_depthconstrained = feature_vector + depth
                
                    feature_vector_depthconstrained = feature_vector_depthconstrained.astype(parent.float)

                    feature_vectors.append(feature_vector_depthconstrained)
    feat = parent.asarray(feature_vectors)

    print(feat.shape)

    return feat

class GenerateFeatureVectors(StagePipeline):
    def __init__(self, window, step, splitlabel=-1, **kwargs):
        super().__init__(name="Generate Feature Vectors", **kwargs)
        
        self.window = window
        self.step = step
        self.splitlabel = splitlabel
        
    def run(self, data, texture, scaler, labels=None):
        if isinstance(data, da.core.Array) and utils.is_executor_cluster(self.dtype):
            minseis = self.xp.min(data)
            maxseis = self.xp.max(data)
            
            data_blocks = da.overlap.overlap(data, depth=(self.step, 0, 0),
                                             boundary='periodic')

            if utils.is_executor_gpu(self.dtype):
                dtype = cp
            else:
                dtype = np
 
            new_data = data_blocks.map_blocks(generate_feature_vectors, texture, self.window, self.step, dtype, minseis, maxseis, labels, self.splitlabel, meta=dtype.array((), dtype=dtype.float32), drop_axis=[1, 2], chunks=(100, 74))

            print("left")

            with get_task_stream(plot='save', filename="scaler-stream.html"):
                new_data = new_data.compute()

            return scaler.fit_transform(new_data)

        elif not utils.is_executor_cluster(self.dtype):
            if isinstance(data, da.core.Array):
                data = data.compute()

            if isinstance(texture, da.core.Array):
                texture = self.xp.asarray(texture.compute())
                
            if labels is not None and isinstance(labels, da.core.Array):
                labels = self.xp.asarray(labels.compute())
            
            minseis = self.xp.min(data)
            maxseis = self.xp.max(data)
            
            if utils.is_executor_gpu(self.dtype):
                dtype = cp
            else:
                dtype = np
            
            new_data = generate_feature_vectors(data, texture, self.window, self.step, dtype, minseis, maxseis, labels, self.splitlabel)
            
            return scaler.fit_transform(new_data)
        
        return data


class PredictStratUnits(StagePipeline):
    def __init__(self, window_size, step, min_cluster_fraction, splitlabel, **kwargs):
        super().__init__(name="Identify Stratigraphy Units", **kwargs)
        
        self.window_size = window_size
        self.step = step
        self.min_cluster_fraction = min_cluster_fraction
        self.splitlabel = splitlabel
        
    def run(self, feature_vectors, seismic, model, labels=None):
        start = datetime.now()

        min_cluster_size = self.xp.int(len(feature_vectors)/self.min_cluster_fraction)

        model.min_cluster_size = min_cluster_size
        
        unique_labels = model.fit_predict(feature_vectors)
        
        datashape = seismic.shape
        
        edge_r = self.xp.int(self.window_size[0]/2 - self.step/2)
        edge_c = self.xp.int(self.window_size[1]/2 - self.step/2)
        edge_d = self.xp.int(self.window_size[2]/2 - self.step/2)

        segmented = self.xp.zeros(datashape, dtype=self.xp.uint64)
        
        if labels is not None and isinstance(labels, da.core.Array):
            with get_task_stream(plot='save', filename="labels-stream.html") as ts:
                labels = self.xp.asarray(labels.compute())
        
        num = 0
        for r in range(0, datashape[0] - self.window_size[0], self.step):
            for c in range(0, datashape[1] - self.window_size[1], self.step):
                for d in range(0, datashape[2] - self.window_size[2], self.step):
                    if labels is not None:
                        window = labels[r+edge_r:r+self.step+edge_r,c+edge_c:c+self.step+edge_c, d+edge_d:d+self.step+edge_d]

                    if self.splitlabel == -1 or labels is None:
                        segmented[r+edge_r:r+self.step+edge_r,c+edge_c:c+self.step+edge_c, d+edge_d:d+self.step+edge_d] = unique_labels[num] + 1
                        num += 1
                    elif window[0, 0, 0] == self.splitlabel:
                        segmented[r+edge_r:r+self.step+edge_r,c+edge_c:c+self.step+edge_c, d+edge_d:d+self.step+edge_d] = unique_labels[num] + 1
                        num += 1
                        
        unique_labels = unique_labels + 1

        end = datetime.now()

        print("Time spent to predict stratigraphy:", end - start)

        return segmented


class ConcatenateSegmentedData(StagePipeline):
    def __init__(self, **kwargs):
        super().__init__(name="Concatenate Clustered Results", **kwargs)
        
    def run(self, intermediate, final):
        if isinstance(intermediate, da.core.Array):
            with get_task_stream(plot='save', filename="intermediate-stream.html") as ts:
                intermediate = intermediate.compute()
            
        if isinstance(intermediate, cp.ndarray):
            intermediate = intermediate.get()
            
        if isinstance(final, da.core.Array):
            with get_task_stream(plot='save', filename="final-stream.html") as ts:
                final = final.compute()
            
        if isinstance(final, cp.ndarray):
            final = final.get()
            
        start = np.max(intermediate)
        
        final = np.where(final > 0, final + start, 0)
         
        final = final + intermediate
        
        del intermediate
        
        return final


if __name__ == '__main__':
    dask = DaskPipelineExecutor(local=True, use_cuda=False, n_workers=4)

    # Data definition
    f3 = F3(chunks={"iline": 350})
    segy2arr = SEGYToArray()

    slice_data = Slice()

    trim = TrimSeismicAmplitudes(99.7)

    lbp3d = LBP3D()

    features = GenerateFeatureVectors([40, 40, 40], 8)
    predict_strat = PredictStratUnits([40, 40, 40], 8, 25, -1, local=True, gpu=False)

    scaler = StantardScaler(local=True, gpu=False)

    hdbscan = HDBSCAN(min_samples=40, local=True, gpu=False)

    # Index that is considered unlabeled
    unlabeled_index = 0

    features_not_labeled = GenerateFeatureVectors([40, 40, 40], 8, unlabeled_index)
    predict_strat_not_labeled = PredictStratUnits([40, 40, 40], 8, 25, unlabeled_index, local=True, gpu=False)

    concat_data = ConcatenateSegmentedData()

    plot_predict_final = Plot2DIline(name="Predicted Data", iline_index=5, cmap="rainbow", swapaxes=(0, 1), filename="plot_iline.png")


    pipeline = ComputePipeline("Stratigraphic Units using HDBSCAN", executor=dask)

    pipeline.add_parameters([f3, scaler, hdbscan])

    pipeline.add(segy2arr, data=f3) \
            .add(slice_data, data=segy2arr) \
            .add(trim, data=slice_data) \
            .add(lbp3d, data=trim) \
            .add(features, data=trim, texture=lbp3d, scaler=scaler) \
            .add(predict_strat, feature_vectors=features, seismic=trim, model=hdbscan) \
            .add(plot_predict_final, data=predict_strat)

    pipeline.visualize(filename="test.png")

#            .add(features_not_labeled, texture=lbp3d, data=trim, scaler=scaler, labels=predict_strat) \
#            .add(predict_strat_not_labeled, feature_vectors=features_not_labeled, seismic=trim, model=hdbscan, labels=predict_strat) \
#            .add(concat_data, intermediate=predict_strat, final=predict_strat_not_labeled) \

    start_time = datetime.now()

    pipeline.run()

    time_elapsed = datetime.now() - start_time 

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
