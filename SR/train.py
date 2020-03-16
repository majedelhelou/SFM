import sys
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
import glob
import cv2
import os
import torch.nn as nn
import math
import argparse

from model.common import save_checkpoint, Hdf5Dataset, adjust_learning_rate
from model.vgg_feature_extractor import VGGFeatureExtractor
from model.GAN import Discriminator_VGG_128, GANLoss
from model.RRDB import RRDBNet
from model.RCAN import RCAN
from model.KMSR import ResBlockNet
from model.IKC import Predictor, Corrector, SFTMD

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--nEpochs", type=int, default=49, help="Number of training epochs")
parser.add_argument("--start_epoch", type=int, default=0, help='Starting Epoch')
parser.add_argument("--net", type=str, default="RCAN", help="RCAN, ESRGAN, RRDB, KMSR")
parser.add_argument("--lr_h5", type=str, default="DIV.h5", help='path of LR h5 file')
parser.add_argument("--hr_h5", type=str, default="None", help='path of HR h5 file')
parser.add_argument("--ngpu", type=int, default=1, help='number of GPUs')
parser.add_argument("--batch_size", type=int, default=16, help='number of GPUs')
parser.add_argument("--resume", type=str, default="", help='restart training checkpoint path')
parser.add_argument("--scale", type=int, default=4, help='scaling factor of SR')

parser.add_argument("--SFM", type=int, default=0, help="0 no SFM, 1 SFM mode for SR")

opt = parser.parse_args()

num_workers = 1
batch_size = opt.batch_size
initial_lr = 0.0001
scale = opt.scale
if (opt.net == 'KMSR'):
    kernel_mode = 'kernel.h5'
else:
    kernel_mode = ''
train_set = Hdf5Dataset(lrname=opt.lr_h5, hrname=opt.hr_h5, sfm=opt.SFM, scale=scale, kernel_mode=kernel_mode)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("===> Building model")

# build model
if (opt.net == 'RCAN'):
    model = RCAN(scale=scale)
elif (opt.net == 'RRDB'):
    model = RRDBNet(scale=scale)
elif (opt.net == 'KMSR'):
    model = ResBlockNet()
elif (opt.net == 'ESRGAN'):
    model = RRDBNet(scale=scale)
    GANcriterion = GANLoss('ragan', 1.0, 0.0)
    l1_factor = 0.01
    mse_factor = 0
    feature_factor = 1.
    gan_factor = 0.005
    gan = Discriminator_VGG_128()
    gan = nn.DataParallel(gan,device_ids=range(opt.ngpu))
    gan.to(device)
    vgg = VGGFeatureExtractor(device=device, feature_layer=34, use_bn=False, use_input_norm=True)
    vgg = nn.DataParallel(vgg,device_ids=range(opt.ngpu))
    vgg.to(device)
    optimizerD = optim.Adam(gan.parameters(), lr=initial_lr, weight_decay=1e-5)
    training_data_loader_gan = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
MSEcriterion = nn.MSELoss()
L1criterion = nn.L1Loss()

    
# resume
model = nn.DataParallel(model, device_ids=range(opt.ngpu))
if (len(opt.resume) > 0):
    model.load_state_dict(torch.load(opt.resume)['model'].state_dict())

model.to(device)
training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

# training
for epoch in range(opt.start_epoch, opt.nEpochs):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    lr = adjust_learning_rate(initial_lr, optimizer, epoch)
    if (opt.net == 'ESRGAN'):
        lr_gan = adjust_learning_rate(initial_lr, optimizerD, epoch)
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        x_data, z_data = Variable(batch[0].float()).cuda(), Variable(batch[1].float()).cuda()
        output = model(z_data)
        
        if (opt.net == 'ESRGAN'):
            x_data_gan, z_data_gan = next(iter(training_data_loader_gan))
            x_data_gan = Variable(x_data_gan.float()).cuda()
            pred_d_real = gan(x_data_gan).detach()
            pred_g_fake = gan(output)
            if opt.net == 'ESRGAN':
                l_g_gan = gan_factor * (
                    GANcriterion(pred_d_real - torch.mean(pred_g_fake), False) +
                    GANcriterion(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            loss = mse_factor * mseloss + l1_factor * loss + feature_factor * L1criterion(vgg(x_data), vgg(output)) + l_g_gan
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (opt.net == 'ESRGAN'):
            optimizerD.zero_grad()
            pred_d_fake = gan(output).detach()
            pred_d_real = gan(x_data_gan)
            l_d_real = GANcriterion(pred_d_real - torch.mean(pred_d_fake), True) * 0.5
            pred_d_fake = gan(output.detach())
            l_d_fake = GANcriterion(pred_d_fake - torch.mean(pred_d_real.detach()), False) * 0.5
            D_loss = l_d_real + l_d_fake
            D_loss.backward()
            optimizerD.step()

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): MSELoss: {:.10f}".format(epoch, iteration, len(training_data_loader), mseloss.item()))
            #save_checkpoint(model, epoch, 'RRDB')
    if (epoch % 10 == 9):
        save_checkpoint('checkpoints/' + opt.net + '/', model, epoch, opt.net)
save_checkpoint('checkpoints/' + opt.net + '/', model, epoch, opt.net)
