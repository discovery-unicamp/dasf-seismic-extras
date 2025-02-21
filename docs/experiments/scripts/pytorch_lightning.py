#!/usr/bin/python3

import os
from os.path import join as pjoin
import collections
import json
import torch
import math
import numbers
import random
import argparse
import itertools
import numpy as np

from typing import List, Dict

import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.strategies import DDPStrategy

from dask_cuda import LocalCUDACluster
from distributed import Client, LocalCluster
from distributed.client import wait, FIRST_COMPLETED
from distributed.utils import TimeoutError as DistributedTimeoutError
import torch.distributed as dist
from torchmetrics import Metric


from sklearn.model_selection import train_test_split

from PIL import Image, ImageOps, ImageChops


class MyAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.idx = 0

    def set_idx(self, idx):
        self.idx = idx

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
#        preds, target = self._input_format(preds, target)
        pred = preds.detach().max(1)[1].cpu().numpy()
        gt = torch.squeeze(target, 1).cpu().numpy()

        assert pred.shape == gt.shape

        np.save("out/pred_" + str(self.idx) + ".npy", pred)

        self.correct += np.sum(pred == gt)
        self.total += len(gt.flatten())

    def __str__(self):
        ret = self.compute()
        return str(ret)

    def compute(self):
        # compute final result
        return float(self.correct / self.total)


def patch_label_2d(model, img, patch_size, stride):
    img = torch.squeeze(img)
    h, w, _ = img.shape  # height and width

    # Pad image with patch_size/2:
    ps = int(np.floor(patch_size/2))  # pad size
    img_p = F.pad(img, pad=(ps, ps, ps, ps), mode='constant', value=0)

    num_classes = 6
    output_p = torch.zeros([1, num_classes, h+2*ps, w+2*ps])

    # generate output:
    for hdx in range(0, h-patch_size+ps, stride):
        for wdx in range(0, w-patch_size+ps, stride):
            patch = img_p[hdx + ps: hdx + ps + patch_size,
                          wdx + ps: wdx + ps + patch_size]
            patch = patch.unsqueeze(dim=0)  # channel dim
            patch = patch.unsqueeze(dim=0)  # batch dim
            
            file_object = open('err.txt', 'w')
            file_object.write(str(patch.shape))
            file_object.close()

            assert (patch.shape == (1, 1, patch_size, patch_size))

            model_output = model(patch)
            output_p[:, :, hdx + ps: hdx + ps + patch_size, wdx + ps: wdx +
                     ps + patch_size] += torch.squeeze(model_output.detach().cpu())

    # crop the output_p in the middke
    output = output_p[:, :, ps:-ps, ps:-ps]
    return output


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):

        img, mask = Image.fromarray(img, mode=None), Image.fromarray(mask, mode='L')
        assert img.size == mask.size

        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask, dtype=np.uint8)

class AddNoise(object):
    def __call__(self, img, mask):
        noise = np.random.normal(loc=0,scale=0.02,size=(img.size[1], img.size[0]))
        return img + noise, mask

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            #Note: we use FLIP_TOP_BOTTOM here intentionaly. Due to the dimensions of the image,
            # it ends up being a horizontal flip.
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask

class RandomVerticallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        '''
        PIL automatically adds zeros to the borders of images that rotated. To fix this
        issue, the code in the botton sets anywhere in the labels (mask) that is zero to
        255 (the value used for ignore_index).
        '''
        rotate_degree = random.random() * 2 * self.degree - self.degree

        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask =  mask.rotate(rotate_degree, Image.NEAREST)

        binary_mask = Image.fromarray(np.ones([mask.size[1], mask.size[0]]))
        binary_mask = binary_mask.rotate(rotate_degree, Image.NEAREST)
        binary_mask = np.array(binary_mask)

        mask_arr = np.array(mask)
        mask_arr[binary_mask==0] = 255
        mask = Image.fromarray(mask_arr)

        return img, mask

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))


class patch_loader(Dataset):
    """
        Data loader for the patch-based deconvnet
    """
    def __init__(self, split='train', stride=50 ,patch_size=99, is_transform=True,
                 augmentations=False):
        self.root = '/data/'
        self.split = split
        self.is_transform = is_transform

        self.augmentations = None
        if augmentations:
            self.augmentations = Compose(
                [RandomRotate(10), RandomHorizontallyFlip(), AddNoise()])

        self.n_classes = 6
        self.mean = 0.000941 # average of the training data
        self.patches = collections.defaultdict(list)
        self.patch_size = patch_size
        self.stride = stride

        if 'test' not in self.split:
            # Normal train/val mode
            self.seismic = self.pad_volume(np.load(pjoin('data','train','train_seismic.npy')))
            self.labels = self.pad_volume(np.load(pjoin('data','train','train_labels.npy')))
        elif 'test1' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test1_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test1_labels.npy'))
        elif 'test2' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test2_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test2_labels.npy'))
        else:
            raise ValueError('Unknown split.')

        if 'test' not in self.split:
            # We are in train/val mode. Most likely the test splits are not saved yet,
            # so don't attempt to load them.
            for split in ['train', 'val', 'train_val']:
                # reading the file names for 'train', 'val', 'trainval'""
                path = pjoin('data', 'splits', 'patch_' + split + '.txt')
                patch_list = tuple(open(path, 'r'))
                patch_list = [id_.rstrip() for id_ in patch_list]
                self.patches[split] = patch_list
        elif 'test' in split:
            # We are in test mode. Only read the given split. The other one might not
            # be available.
            path = pjoin('data', 'splits', 'patch_' + split + '.txt')
            patch_list = tuple(open(path,'r'))
            patch_list = [id_.rstrip() for id_ in patch_list]
            self.patches[split] = patch_list
        else:
            raise ValueError('Unknown split.')

    def pad_volume(self,volume):
        '''
        Only used for train/val!! Not test.
        '''
        assert 'test' not in self.split, 'There should be no padding for test time!'
        return np.pad(volume,pad_width=self.patch_size,mode='constant', constant_values=255)


    def __len__(self):
        return len(self.patches[self.split])

    def __getitem__(self, index):

        patch_name = self.patches[self.split][index]
        direction, idx, xdx, ddx = patch_name.split(sep='_')

        shift = (self.patch_size if 'test' not in self.split else 0)
        idx, xdx, ddx = int(idx)+shift, int(xdx)+shift, int(ddx)+shift

        if direction == 'i':
            im = self.seismic[idx,xdx:xdx+self.patch_size,ddx:ddx+self.patch_size]
            lbl = self.labels[idx,xdx:xdx+self.patch_size,ddx:ddx+self.patch_size]
        elif direction == 'x':
            im = self.seismic[idx: idx+self.patch_size, xdx, ddx:ddx+self.patch_size]
            lbl = self.labels[idx: idx+self.patch_size, xdx, ddx:ddx+self.patch_size]

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        img -= self.mean

        # to be in the BxCxHxW that PyTorch uses:
        img, lbl = img.T, lbl.T

        img = np.expand_dims(img,0)
        lbl = np.expand_dims(lbl,0)

        img = torch.from_numpy(img)
        img = img.float()
        lbl = torch.from_numpy(lbl)
        lbl = lbl.long()

        return img, lbl

    def get_seismic_labels(self):
        return np.asarray([ [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89],
                          [215,48,39]])


    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_seismic_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
            
            
class section_loader(Dataset):
    """
        Data loader for the section-based deconvnet
    """
    def __init__(self, split='train', is_transform=True,
                 augmentations=False):
        self.root = 'data/'
        self.split = split
        self.is_transform = is_transform
        self.augmentations = None
        if augmentations:
            self.augmentations = Compose(
                [RandomRotate(10), RandomHorizontallyFlip(), AddNoise()])
        self.n_classes = 6 
        self.mean = 0.000941 # average of the training data  
        self.sections = collections.defaultdict(list)

        if 'test' not in self.split: 
            # Normal train/val mode
            self.seismic = np.load(pjoin('data','train','train_seismic.npy'))
            self.labels = np.load(pjoin('data','train','train_labels.npy'))
        elif 'test1' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test1_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test1_labels.npy'))
        elif 'test2' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test2_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test2_labels.npy'))
        else:
            raise ValueError('Unknown split.')

        if 'test' not in self.split:
            # We are in train/val mode. Most likely the test splits are not saved yet, 
            # so don't attempt to load them.  
            for split in ['train', 'val', 'train_val']:
                # reading the file names for 'train', 'val', 'trainval'""
                path = pjoin('data', 'splits', 'section_' + split + '.txt')
                file_list = tuple(open(path, 'r'))
                file_list = [id_.rstrip() for id_ in file_list]
                self.sections[split] = file_list
        elif 'test' in split:
            # We are in test mode. Only read the given split. The other one might not 
            # be available. 
            path = pjoin('data', 'splits', 'section_' + split + '.txt')
            file_list = tuple(open(path,'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.sections[split] = file_list
        else:
            raise ValueError('Unknown split.')


    def __len__(self):
        return len(self.sections[self.split])

    def __getitem__(self, index):

        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep='_')

        if direction == 'i':
            im = self.seismic[int(number),:,:]
            lbl = self.labels[int(number),:,:]
        elif direction == 'x':    
            im = self.seismic[:,int(number),:]
            lbl = self.labels[:,int(number),:]
        
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
            
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


    def transform(self, img, lbl):
        img -= self.mean

        # to be in the BxCxHxW that PyTorch uses: 
        img, lbl = img.T, lbl.T

        img = np.expand_dims(img,0)
        lbl = np.expand_dims(lbl,0)

        img = torch.from_numpy(img)
        img = img.float()
        lbl = torch.from_numpy(lbl)
        lbl = lbl.long()
                
        return img, lbl

    def get_seismic_labels(self):
        return np.asarray([ [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89],
                          [215,48,39]])


    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_seismic_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


class section_deconvnet(pl.LightningModule):

    def __init__(self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False, batch_size=1):
        super().__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.clip = clip
        self.batch_size = batch_size

        if class_weights:
            self.class_weights = torch.tensor(
                [0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], requires_grad=False)
        else:
            self.class_weights = None

        self.class_names = ['upper_ns', 'middle_ns', 'lower_ns',
                            'rijnland_chalk', 'scruff', 'zechstein']

        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv_block1 = nn.Sequential(

            # conv1_1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv1_2
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool1
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_1

        # 48*48

        self.conv_block2 = nn.Sequential(

            # conv2_1
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv2_2
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool2
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_2

        # 24*24

        self.conv_block3 = nn.Sequential(

            # conv3_1
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv3_2
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv3_3
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool3
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_3

        # 12*12

        self.conv_block4 = nn.Sequential(

            # conv4_1
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv4_2
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv4_3
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool4
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_4

        # 6*6

        self.conv_block5 = nn.Sequential(

            # conv5_1
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv5_2
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv5_3
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool5
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_5

        # 3*3

        self.conv_block6 = nn.Sequential(

            # fc6
            nn.Conv2d(512, 4096, 3),
            # set the filter size and nor padding to make output into 1*1
            nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        # 1*1

        self.conv_block7 = nn.Sequential(

            # fc7
            nn.Conv2d(4096, 4096, 1),
            # set the filter size to make output into 1*1
            nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.deconv_block8 = nn.Sequential(

            # fc6-deconv
            nn.ConvTranspose2d(4096, 512, 3, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        # 3*3

        self.unpool_block9 = nn.Sequential(

            # unpool5
            nn.MaxUnpool2d(2, stride=2), )
        # usage unpool(output, indices)

        # 6*6

        self.deconv_block10 = nn.Sequential(

            # deconv5_1
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv5_2
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv5_3
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block11 = nn.Sequential(

            # unpool4
            nn.MaxUnpool2d(2, stride=2), )

        # 12*12

        self.deconv_block12 = nn.Sequential(

            # deconv4_1
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv4_2
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv4_3
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block13 = nn.Sequential(

            # unpool3
            nn.MaxUnpool2d(2, stride=2), )

        # 24*24

        self.deconv_block14 = nn.Sequential(

            # deconv3_1
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv3_2
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv3_3
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block15 = nn.Sequential(

            # unpool2
            nn.MaxUnpool2d(2, stride=2), )

        # 48*48

        self.deconv_block16 = nn.Sequential(

            # deconv2_1
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv2_2
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block17 = nn.Sequential(

            # unpool1
            nn.MaxUnpool2d(2, stride=2), )

        # 96*96

        self.deconv_block18 = nn.Sequential(

            # deconv1_1
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv1_2
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.seg_score19 = nn.Sequential(

            # seg-score
            nn.Conv2d(64, self.n_classes, 1), )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        size0 = x.size()
        conv1, indices1 = self.conv_block1(x)
        size1 = conv1.size()
        conv2, indices2 = self.conv_block2(conv1)
        size2 = conv2.size()
        conv3, indices3 = self.conv_block3(conv2)
        size3 = conv3.size()
        conv4, indices4 = self.conv_block4(conv3)
        size4 = conv4.size()
        conv5, indices5 = self.conv_block5(conv4)

        conv6 = self.conv_block6(conv5)
        conv7 = self.conv_block7(conv6)
        conv8 = self.deconv_block8(conv7) 
        conv9 = self.unpool(conv8,indices5, output_size=size4)
        conv10 = self.deconv_block10(conv9) 
        conv11 = self.unpool(conv10,indices4, output_size=size3)
        conv12 = self.deconv_block12(conv11) 
        conv13 = self.unpool(conv12,indices3, output_size=size2)
        conv14 = self.deconv_block14(conv13) 
        conv15 = self.unpool(conv14,indices2, output_size=size1)
        conv16 = self.deconv_block16(conv15) 
        conv17 = self.unpool(conv16,indices1, output_size=size0)
        conv18 = self.deconv_block18(conv17)
        out = self.seg_score19(conv18)

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        i_layer = 0;
        # copy convolutional filters from vgg16
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    if i_layer == 0:
                        l2.weight.data = ((l1.weight.data[:, 0, :, :] + l1.weight.data[:, 1, :, :] + l1.weight.data[:,
                                                                                                     2, :,
                                                                                                     :]) / 3.0).view(
                            l2.weight.size())
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
                    else:
                        assert l1.weight.size() == l2.weight.size()
                        assert l1.bias.size() == l2.bias.size()
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
                        
    def cross_entropy_loss(self, input, target, weight=None, ignore_index=255):
        '''
        Use 255 to fill empty values when padding or doing any augmentation operations
        like rotation.
        '''
        target = torch.squeeze(target, dim=1)
        loss = F.cross_entropy(input, target, weight, reduction='sum',  ignore_index=255)

        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.forward(images)

        loss = self.cross_entropy_loss(input=outputs, target=labels, weight=self.class_weights)

        # gradient clipping
        if self.clip != 0:
             nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        return loss

    def prepare_data(self):
        self.train_set = section_loader(is_transform=True,
                                        split='train',
                                        augmentations=True)

        self.val_set = section_loader(is_transform=True,
                                      split='val')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
        )
        return val_loader

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch

        outputs = self.forward(images)

        loss = self.cross_entropy_loss(input=outputs, target=labels)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
        return optimizer


class patch_deconvnet(pl.LightningModule):

    def __init__(self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False, batch_size=1):
        super().__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.clip = clip
        self.batch_size = batch_size
        self.accuracy = MyAccuracy()

        if class_weights:
            self.class_weights = torch.tensor(
                [0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], requires_grad=False)
        else:
            self.class_weights = None

        self.class_names = ['upper_ns', 'middle_ns', 'lower_ns',
                            'rijnland_chalk', 'scruff', 'zechstein']

        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv_block1 = nn.Sequential(

            # conv1_1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv1_2
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool1
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_1

        # 48*48

        self.conv_block2 = nn.Sequential(

            # conv2_1
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv2_2
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool2
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_2

        # 24*24

        self.conv_block3 = nn.Sequential(

            # conv3_1
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv3_2
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv3_3
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool3
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_3

        # 12*12

        self.conv_block4 = nn.Sequential(

            # conv4_1
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv4_2
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv4_3
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool4
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_4

        # 6*6

        self.conv_block5 = nn.Sequential(

            # conv5_1
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv5_2
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv5_3
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool5
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_5

        # 3*3

        self.conv_block6 = nn.Sequential(

            # fc6
            nn.Conv2d(512, 4096, 3),
            # set the filter size and nor padding to make output into 1*1
            nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        # 1*1

        self.conv_block7 = nn.Sequential(

            # fc7
            nn.Conv2d(4096, 4096, 1),
            # set the filter size to make output into 1*1
            nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.deconv_block8 = nn.Sequential(

            # fc6-deconv
            nn.ConvTranspose2d(4096, 512, 3, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        # 3*3

        self.unpool_block9 = nn.Sequential(

            # unpool5
            nn.MaxUnpool2d(2, stride=2), )
        # usage unpool(output, indices)

        # 6*6

        self.deconv_block10 = nn.Sequential(

            # deconv5_1
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv5_2
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv5_3
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block11 = nn.Sequential(

            # unpool4
            nn.MaxUnpool2d(2, stride=2), )

        # 12*12

        self.deconv_block12 = nn.Sequential(

            # deconv4_1
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv4_2
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv4_3
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block13 = nn.Sequential(

            # unpool3
            nn.MaxUnpool2d(2, stride=2), )

        # 24*24

        self.deconv_block14 = nn.Sequential(

            # deconv3_1
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv3_2
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv3_3
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block15 = nn.Sequential(

            # unpool2
            nn.MaxUnpool2d(2, stride=2), )

        # 48*48

        self.deconv_block16 = nn.Sequential(

            # deconv2_1
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv2_2
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block17 = nn.Sequential(

            # unpool1
            nn.MaxUnpool2d(2, stride=2), )

        # 96*96

        self.deconv_block18 = nn.Sequential(

            # deconv1_1
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv1_2
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.seg_score19 = nn.Sequential(

            # seg-score
            nn.Conv2d(64, self.n_classes, 1), )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        size0 = x.size()
        conv1, indices1 = self.conv_block1(x)
        size1 = conv1.size()
        conv2, indices2 = self.conv_block2(conv1)
        size2 = conv2.size()
        conv3, indices3 = self.conv_block3(conv2)
        size3 = conv3.size()
        conv4, indices4 = self.conv_block4(conv3)
        size4 = conv4.size()
        conv5, indices5 = self.conv_block5(conv4)

        conv6 = self.conv_block6(conv5)
        conv7 = self.conv_block7(conv6)
        conv8 = self.deconv_block8(conv7)
        conv9 = self.unpool(conv8, indices5, output_size=size4)
        conv10 = self.deconv_block10(conv9)
        conv11 = self.unpool(conv10, indices4, output_size=size3)
        conv12 = self.deconv_block12(conv11)
        conv13 = self.unpool(conv12, indices3, output_size=size2)
        conv14 = self.deconv_block14(conv13)
        conv15 = self.unpool(conv14, indices2, output_size=size1)
        conv16 = self.deconv_block16(conv15)
        conv17 = self.unpool(conv16, indices1, output_size=size0)
        conv18 = self.deconv_block18(conv17)
        out = self.seg_score19(conv18)

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        i_layer = 0;
        # copy convolutional filters from vgg16
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    if i_layer == 0:
                        l2.weight.data = ((l1.weight.data[:, 0, :, :] + l1.weight.data[:, 1, :, :] + l1.weight.data[:,
                                                                                                     2, :,
                                                                                                     :]) / 3.0).view(
                            l2.weight.size())
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
                    else:
                        assert l1.weight.size() == l2.weight.size()
                        assert l1.bias.size() == l2.bias.size()
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1

    def cross_entropy_loss(self, input, target, weight=None, ignore_index=255):
        '''
        Use 255 to fill empty values when padding or doing any augmentation operations
        like rotation.
        '''
        target = torch.squeeze(target, dim=1)
        loss = F.cross_entropy(input, target, weight, reduction='sum',  ignore_index=255)

        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.forward(images)

        loss = self.cross_entropy_loss(input=outputs, target=labels, weight=self.class_weights)

        # gradient clipping
        if self.clip != 0:
             nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        return loss

    def prepare_data(self):
        self.train_set = patch_loader(is_transform=True,
                                      split='train',
                                      stride=50,
                                      patch_size=99,
                                      augmentations=True)

        self.val_set = patch_loader(is_transform=True,
                                    split='val',
                                    stride=50,
                                    patch_size=99)
                                    
        self.test1_set = section_loader(is_transform=True,
                                        split="test1",
                                        augmentations=None)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
        )
        return val_loader
        
    def test_dataloader(self):
        test1_loader = DataLoader(
            self.test1_set,
            batch_size=self.batch_size
        )
        return test1_loader
        
    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
 
        preds = self(images)
        
        file_object = open('test.txt', 'w')
        file_object.write(str(preds.shape))
        file_object.write(str(labels.shape))
        file_object.close()
#        outputs = patch_label_2d(model=self,
#                                 img=images,
#                                 patch_size=99,
#                                 stride=10)
                                 
#        pred = outputs.detach().max(1)[1].numpy()
#        gt = labels.numpy()

        self.accuracy.set_idx(batch_idx)
        
        self.accuracy(preds, labels)
        
    def test_step_end(self, output_results):
        file_object = open('acc.txt', 'w')
        file_object.write(str(self.accuracy))
        file_object.close()

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch

        outputs = self.forward(images)

        loss = self.cross_entropy_loss(input=outputs, target=labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), amsgrad=True)
        return optimizer
        
        
def split_train_val_patch(stride=50, per_val=0.2):
    loader_type='patch'
    # create inline and crossline pacthes for training and validation:
    labels = np.load(pjoin('data', 'train', 'train_labels.npy'))
    iline, xline, depth = labels.shape

    # INLINE PATCHES: ------------------------------------------------
    i_list = []
    horz_locations = range(0, xline-stride, stride)
    vert_locations = range(0, depth-stride, stride)
    for i in range(iline):
        # for every inline:
        # images are references by top-left corner:
        locations = [[j, k] for j in horz_locations for k in vert_locations]
        patches_list = ['i_'+str(i)+'_'+str(j)+'_'+str(k)
                        for j, k in locations]
        i_list.append(patches_list)

    # flatten the list
    i_list = list(itertools.chain(*i_list))

    # XLINE PATCHES: ------------------------------------------------
    x_list = []
    horz_locations = range(0, iline-stride, stride)
    vert_locations = range(0, depth-stride, stride)
    for j in range(xline):
        # for every xline:
        # images are references by top-left corner:
        locations = [[i, k] for i in horz_locations for k in vert_locations]
        patches_list = ['x_'+str(i)+'_'+str(j)+'_'+str(k)
                        for i, k in locations]
        x_list.append(patches_list)

    # flatten the list
    x_list = list(itertools.chain(*x_list))

    list_train_val = i_list + x_list

    # create train and test splits:
    list_train, list_val = train_test_split(
        list_train_val, test_size=per_val, shuffle=True)

    os.makedirs(pjoin("data", "splits"), exist_ok=True)
    
    os.makedirs(pjoin('data', 'splits'), exist_ok=True)

    # write to files to disK:
    file_object = open(
        pjoin('data', 'splits', loader_type + '_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(
        pjoin('data', 'splits', loader_type + '_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(pjoin('data', 'splits', loader_type + '_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()
    
    
def split_train_val_section(per_val=0.1):
    # create inline and crossline sections for training and validation:
    loader_type = 'section'
    labels = np.load(pjoin('data', 'train', 'train_labels.npy'))
    i_list = list(range(labels.shape[0]))
    i_list = ['i_'+str(inline) for inline in i_list]

    x_list = list(range(labels.shape[1]))
    x_list = ['x_'+str(crossline) for crossline in x_list]

    list_train_val = i_list + x_list

    # create train and test splits:
    list_train, list_val = train_test_split(
        list_train_val, test_size=per_val, shuffle=True)
        
    os.makedirs(pjoin('data', 'splits'), exist_ok=True)

    # write to files to disK:
    file_object = open(
        pjoin('data', 'splits', loader_type + '_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(
        pjoin('data', 'splits', loader_type + '_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(pjoin('data', 'splits', loader_type + '_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()
    
    
def split_test_section(split="test1"):
    labels = np.load(pjoin('data', 'test_once', split + '_labels.npy'))
    irange, xrange, depth = labels.shape

    i_list = list(range(irange))
    i_list = ['i_'+str(inline) for inline in i_list]

    x_list = list(range(xrange))
    x_list = ['x_'+str(crossline) for crossline in x_list]

    list_test = i_list# + x_list

    file_object = open(
        pjoin('data', 'splits', 'section_' + split + '.txt'), 'w')
    file_object.write('\n'.join(list_test))
    file_object.close()
    
    
def split_test_patch(split="test1", per_val=0.1, stride=50):
    loader_type = 'patch'
    labels = np.load(pjoin('data', 'test_once', split + '_labels.npy'))
    iline, xline, depth = labels.shape

    # INLINE PATCHES: ------------------------------------------------
    i_list = []
    horz_locations = range(0, xline-stride, stride)
    vert_locations = range(0, depth-stride, stride)
    for i in range(iline):
        # for every inline:
        # images are references by top-left corner:
        locations = [[j, k] for j in horz_locations for k in vert_locations]
        patches_list = ['i_'+str(i)+'_'+str(j)+'_'+str(k)
                        for j, k in locations]
        i_list.append(patches_list)

    # flatten the list
    i_list = list(itertools.chain(*i_list))

    # XLINE PATCHES: ------------------------------------------------
    x_list = []
    horz_locations = range(0, iline-stride, stride)
    vert_locations = range(0, depth-stride, stride)
    for j in range(xline):
        # for every xline:
        # images are references by top-left corner:
        locations = [[i, k] for i in horz_locations for k in vert_locations]
        patches_list = ['x_'+str(i)+'_'+str(j)+'_'+str(k)
                        for i, k in locations]
        x_list.append(patches_list)

    # flatten the list
    x_list = list(itertools.chain(*x_list))

    list_test = i_list + x_list

    # write to files to disK:
    file_object = open(
        pjoin('data', 'splits', 'patch_' + split + '.txt'), 'w')
    file_object.write('\n'.join(list_test))
    file_object.close()


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, batch_size=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

        self.batch_size = batch_size

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def prepare_data(self):
        dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
        self.train_set, self.val_set = random_split(dataset, [55000, 5000])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
#            ampler=self.sampler,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
        )
        return val_loader


def _get_worker_info(client: Client) -> List[Dict]:
    """
    returns a list of workers (sorted), and the DNS name for the master host
    The master is the 0th worker's host
    """
    workers = client.scheduler_info()["workers"]
    worker_keys = sorted(workers.keys())
    workers_by_host: Dict[str, List[str]] = {}
    for key in worker_keys:
        worker = workers[key]
        host = worker["host"]
        workers_by_host.setdefault(host, []).append(key)
    host = workers[worker_keys[0]]["host"]
    all_workers = []
    global_rank = 0
    world_size = len(workers_by_host)
    hosts = sorted(workers_by_host.keys())
    for host in hosts:
        local_rank = 0
        for worker in workers_by_host[host]:
            all_workers.append(
                dict(
                    master=hosts[0],
                    worker=worker,
                    local_rank=0,
                    global_rank=global_rank,
                    host=host,
                    world_size=world_size,
                )
            )
            local_rank += 1
            global_rank += 1
    return all_workers


def train_test(metadata, model, accel, strategy, batch_size, use_mnist):
    class MyClusterEnvironment(ClusterEnvironment):
        def __init__(self, metadata) -> None:
            super().__init__()
            self.metadata = metadata
            self._master_port = 23456

        @staticmethod
        def detect() -> bool:
            if "master" not in self.metadata:
                return False
            if "world_size" not in self.metadata:
                return False
            if "global_rank" not in self.metadata:
                return False

            return True

        @property
        def creates_processes_externally(self) -> bool:
            """Return True if the cluster is managed (you don't launch processes yourself)"""
            return True

        def creates_children(self) -> bool:
            return False

        def world_size(self) -> int:
            return int(self.metadata["world_size"])

        def global_rank(self) -> int:
            return int(self.metadata["global_rank"])

        def local_rank(self) -> int:
            if "local_rank" in self.metadata:
                return int(self.metadata["local_rank"])
            else:
                return 0

        def node_rank(self) -> int:
            return int(self.metadata["global_rank"])

        @property
        def main_address(self) -> str:
            print(self.metadata)
            return self.metadata["master"]

        @property
        def main_port(self) -> int:
            return self._master_port

        def set_world_size(self, size: int) -> None:
            pass

        def set_global_rank(self, rank: int) -> None:
            pass

    if not use_mnist:
        split_train_val_section()
        split_train_val_patch()
        split_test_section()
        split_test_patch()

    gpus = 0
    if accel == "gpu":
        gpus = -1

    auto_batch_size = None
    if batch_size < 0:
        auto_batch_size = "binsearch"

    trainer = pl.Trainer(max_epochs=100, accelerator=accel, strategy=strategy, plugins=[MyClusterEnvironment(metadata)],
                         gpus=gpus, num_nodes=metadata["world_size"], profiler="simple", auto_scale_batch_size=auto_batch_size)

    if auto_batch_size is not None:
        trainer.tune(model)

    trainer.fit(model)
    
    trainer.test(model)


def main(client=None, accel="cpu", strategy="dp", num_nodes=1, num_gpus=0, batch_size=32, use_mnist=True):
    if not use_mnist:
        model = patch_deconvnet(n_classes=6, batch_size=batch_size)
    else:
        model = LitAutoEncoder(batch_size)

    if client is not None:
        all_workers = _get_worker_info(client)

        for worker in all_workers:
            futures = client.submit(train_test, worker, model, accel, strategy, batch_size, use_mnist, workers=[worker["worker"]])

        while True:
            if not futures:
                break

            try:
                result = wait(futures, 0.1, FIRST_COMPLETED)
            except DistributedTimeoutError:
                continue

            for fut in result.done:
                try:
                    fut.result(timeout=14400)
                except Exception as e:  # pylint: disable=broad-except
                    print(str(e))
                    raise
            futures = result.not_done
    else:
        split_train_val_patch()
        split_train_val_section()
        split_test_section()
        split_test_patch()

        gpus = num_gpus
        if accel != "gpu":
            gpus = 0

        auto_batch_size = None
        if batch_size < 0:
            auto_batch_size = "binsearch"

        trainer = pl.Trainer(max_epochs=100, accelerator=accel, strategy=strategy, gpus=gpus, num_nodes=num_nodes,
                             profiler="simple", auto_scale_batch_size=auto_batch_size)

        if auto_batch_size is not None:
            trainer.tune(model)

        trainer.fit(model)
        
        trainer.test(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infra Parameters')
    parser.add_argument('--address', nargs='?', type=str, default=None,
                        help='Dask Cluster Address for multi arch environments')
    parser.add_argument('--local-cluster', nargs='?', type=bool, default=False,
                        const=True, help='Spawn a local cluster')
    parser.add_argument('--use-cuda', nargs='?', type=bool, default=False,
                        const=True, help='Use a CUDA cluster if it is local or not')
    parser.add_argument('--use-mnist', nargs='?', type=bool, default=False,
                        const=True, help='Use MNIST benchmark instead')
    parser.add_argument('--num-nodes', nargs='?', type=int, default=1,
                        help='Number of distributed nodes')
    parser.add_argument('--num-gpus', nargs='?', type=int, default=0,
                        help='Number of GPUs of each node')
    parser.add_argument('--strategy', nargs='?', type=str, default=None,
                        help='Trainer strategy')
    parser.add_argument('--batch-size', nargs='?', type=int, default=32,
                        help='Data Loader batch size. If you pass -1 it means auto size batch.')


    args = parser.parse_args()

    local_cluster = args.local_cluster
    if args.address and args.local_cluster:
        print("Warning: `--address` and `--local-cluster` are opposite commands. Using `--address` instead.")
        local_cluster = False

    client = None
    if args.address or local_cluster:
        strategy = "ddp"
        if args.address:
            client = Client(args.address)
        elif local_cluster:
            if args.use_cuda:
                cluster = LocalCUDACluster()
                client = Client(cluster)
            else:
                cluster = LocalCluster()
                client = Client(cluster)
    elif args.strategy:
        strategy = args.strategy
    else:
        strategy = "dp"

    if args.use_cuda:
        accel = "gpu"
    else:
        accel = "cpu"

    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)

    main(client, accel, strategy, args.num_nodes, args.num_gpus, args.batch_size, args.use_mnist)
