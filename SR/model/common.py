import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import functools
from functools import partial
import numpy as np
import random
from imageio import imread
import glob
import cv2
import os
import torch.nn as nn
import math
from utils_SFM import *
from scipy import signal


def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 10))
    return lr

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer

def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding

def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, \
                act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)

def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,\
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# checkpoint
def save_checkpoint(model_dir, model, epoch, name):
    model_out_path = "%s/epoch_%d.pth" % (model_dir, epoch)
    state = {"epoch": epoch ,"model": model, "name": name}
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, :, ::-1]
        if vflip: img = img[:, ::-1, :]
        if rot90: img = img.transpose(0, 2, 1)
        
        return img

    return [_augment(_l) for _l in l]    
    
# dataset
class Hdf5Dataset(data.Dataset):
    def __init__(self, lrname='.', hrname='', sfm=0, scale=4, kernel_mode=''):
        super(Hdf5Dataset, self).__init__()      
        self.hr_dataset = h5py.File(hrname, 'r')['/data']
        if not lrname == 'None':
            self.lr_dataset = h5py.File(lrname, 'r')['/data']
        else:
            self.lr_dataset = 'None'
        
        if (kernel_mode == 'KMSR'):
            self.kernel = h5py.File(kernel_mode, 'r')['/data']
        self.sfm = sfm

    def __getitem__(self, index):
        hr_img = (self.hr_dataset[index])
        if (lr_dataset == 'None'):
            lr_img = self.hr_dataset[index]
        else:
            lr_img = (self.lr_dataset[index])
        
        # data augmentation
        [hr_img, lr_img] = augment([hr_img, lr_img], True, True)
        
        # SFM
        if (sfm > 0 and random.random() > 0.5):
            lr_img, mask = random_drop(lr_img, mode=0)
        if (lr_dataset != 'None'):
            lr_img = cv2.imresize(lr_img, (0,0), fx=1/scale, fy=1/scale)
        
        # KMSR: kernel blur
        kernel_index = min(random.randint(0,1999), len(self.kernel)-1)
        kernel = self.kernel[kernel_index]
        lr_img[0,:,:] = signal.convolve2d(lr_img[0,:,:], kernel[0,:,:], 'same')
        lr_img[1,:,:] = signal.convolve2d(lr_img[1,:,:], kernel[0,:,:], 'same')
        lr_img[2,:,:] = signal.convolve2d(lr_img[2,:,:], kernel[0,:,:], 'same')
            
        return hr_img.astype('float'), lr_img.astype('float')

    def __len__(self):
        return self.hr_dataset.shape[0]

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
