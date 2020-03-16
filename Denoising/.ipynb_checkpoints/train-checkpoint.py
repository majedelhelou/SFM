import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from models import get_model_name, DnCNN_RL, BUIFD, MemNet
from models import RIDNET as RIDNet
from dataset import prepare_data, Dataset
from utils import *
from utils_SFM import *
import time
from scipy.fftpack import dct, idct
import torch_dct as torch_dct
from scipy.ndimage.filters import gaussian_filter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="SFM")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--weight_exponent", type=float, default=0, help="Exponent of noise level loss weight")
parser.add_argument("--color", type=int, default=0, help='1 for a color network, 0 for grayscale')

parser.add_argument("--DCT_DOR", type=float, default=0, help="DCT Dropout Rate, if 0 no DCT dropout")
parser.add_argument("--SFM_mode", type=int, default=0, help="0 fully random, 1 circular, 2 sweeping")
parser.add_argument("--SFM_rad_perc", type=float, default=0.5, help="IN MODE 2: the central band for SFM is SFM_rad_perc*max_radius")
parser.add_argument("--SFM_sigma_perc", type=float, default=0.05, help="IN MODE 2: the stddev around the central band for SFM")
parser.add_argument("--SFM_noise", type=int, default=0, help="0: SFM input before adding noise, 1 after")
parser.add_argument("--SFM_GT", type=int, default=0, help="0: don't SFM the GT, 1 SFM GT with input's mask")
parser.add_argument("--mask_train_noise", type=int, default=0, help='if 0: nothing; if 1: high frequency noise only; if 2: low frequency noise only; if 3: Brownian noise')

parser.add_argument("--DCTloss_weight", type=int, default=0, help='if 0: nothing; if 1: adds auxiliary DCT-reconstruction loss')

parser.add_argument("--net_mode", type=str, default="R", help='DnCNN_RL (R), MemNet (M) or RIDNet (D)')
parser.add_argument("--noise_max", type=float, default=55, help="Max training noise level")
opt = parser.parse_args()


def main():

    # Load dataset
    print('Loading dataset ...\n')
    ## R, M, D ##
    if opt.net_mode == 'R' or opt.net_mode == 'M' or opt.net_mode == 'D':
        if opt.color == 1:
            dataset_train = Dataset(train=True, aug_times=2, grayscale=False, scales=True)
        else:
            dataset_train = Dataset(train=True, aug_times=2, grayscale=True, scales=True)
    else:
        raise NotImplemented('Supported networks: R (DnCNN), M (MemNet), D (RIDNet) only')


    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))


    # Build model
    model_channels = 1 + 2 * opt.color

    np.random.seed(0)
    torch.manual_seed(0)
    
    if opt.net_mode == 'R':
        print('** Creating DnCNN RL network: **')
        net = DnCNN_RL(channels=model_channels, num_of_layers=opt.num_of_layers)
    elif opt.net_mode == 'M':
        print('** Creating MemNet network: **')
        net = MemNet(in_channels=model_channels, channels=20, num_memblock=6, num_resblock=4)
    elif opt.net_mode == 'D':
        print('** Creating RIDNet network: **')
        net = RIDNet(in_channels=model_channels)
    print(net)
    net.apply(weights_init_kaiming)
    

    # Loss
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    model = nn.DataParallel(net).cuda()
    criterion.cuda() # print(model)
    print('Trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Training
    noiseL_B = [0,opt.noise_max]

    train_loss_log = np.zeros(opt.epochs)
    
    for epoch in range(opt.epochs):

        # Learning rate
        factor = epoch // opt.milestone
        current_lr = opt.lr / (10.**factor)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('\nlearning rate %f' % current_lr)

        # Train
        t = time.time()
        for i, data in enumerate(loader_train, 0):
            # Training
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            
            # ADD Noise
            img_train = data
                        
            noise = torch.zeros(img_train.size())
            noise_level_train = torch.zeros(img_train.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
            sizeN = noise[0,:,:,:].size()
                        
            # Noise Level map preparation (each step)
            for n in range(noise.size()[0]):
                noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
                noise_level_value = stdN[n] / noiseL_B[1]
                noise_level_train[n,:,:,:] = torch.FloatTensor( np.ones(sizeN) )
                noise_level_train[n,:,:,:] = noise_level_train[n,:,:,:] * noise_level_value
            noise_level_train = Variable(noise_level_train.cuda())

            # Modifying the frequency content of the added noise (Low or High only)
            if opt.mask_train_noise in([1,2]):
                noise_mask = get_mask_low_high(w=sizeN[1], h=sizeN[2], radius_perc=0.5, mask_mode=opt.mask_train_noise)
                for n in range(noise.size()[0]):
                    noise_dct = dct(dct(noise[n,0,:,:].data.numpy(), axis=0, norm='ortho'), axis=1, norm='ortho')
                    noise_dct = noise_dct * noise_mask
                    noise_numpy = idct(idct(noise_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
                    noise[n,0,:,:] = torch.from_numpy(noise_numpy)
            elif opt.mask_train_noise == 3: #Brownian noise
                for n in range(noise.size()[0]):
                    noise_numpy = gaussian_filter(noise[n,0,:,:].data.numpy(), sigma=3)
                    noise[n,0,:,:] = torch.from_numpy(noise_numpy)
            
            
            # DCT SFM
            if opt.DCT_DOR > 0:
                img_train_SFM = np.zeros(img_train.size(),dtype='float32')
                noise_SFM = np.zeros(noise.size(),dtype='float32')
                
                dct_bool = np.random.choice([1, 0], size=(img_train.size()[0],), p=[opt.DCT_DOR, 1-opt.DCT_DOR])
                for img_idx in range(img_train.size()[0]):
                    if dct_bool[img_idx] == 1:
                        img_numpy, mask = random_drop(img_train[img_idx,:,:,:].data.numpy(), mode=opt.SFM_mode, SFM_center_radius_perc=opt.SFM_rad_perc, SFM_center_sigma_perc=opt.SFM_sigma_perc)
                        img_train_SFM[img_idx,0,:,:] = img_numpy
                        
                        noise_dct = dct(dct(noise[img_idx,0,:,:].data.numpy(), axis=0, norm='ortho'), axis=1, norm='ortho')
                        noise_dct = noise_dct * mask
                        noise_numpy = idct(idct(noise_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
                        noise_SFM[img_idx,0,:,:] = noise_numpy

                if opt.SFM_noise == 0:
                    imgn_train = torch.from_numpy(img_train_SFM) + noise
                elif opt.SFM_noise == 1:
                    imgn_train = torch.from_numpy(img_train_SFM) + torch.from_numpy(noise_SFM)

                if opt.SFM_GT == 1:
                    img_train = torch.from_numpy(img_train_SFM)
            else:
                imgn_train = img_train + noise
                


            # Training step
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            
            out_train = model(imgn_train)
            OUT_NOISE = imgn_train-out_train
            loss = criterion(OUT_NOISE, noise) / (imgn_train.size()[0]*2)

            if (opt.DCTloss_weight==1):
                noise_DCT = torch.zeros(noise.size())
                OUT_NOISE_DCT = torch.zeros(noise.size())
                for img_idx in range(noise.size()[0]):
                    noise_DCT[img_idx,0,:,:] = torch_dct.dct_2d(noise[img_idx,0,:,:])
                    OUT_NOISE_DCT[img_idx,0,:,:] = torch_dct.dct_2d(OUT_NOISE[img_idx,0,:,:])
                noise_DCT, OUT_NOISE_DCT = Variable(noise_DCT.cuda()), Variable(OUT_NOISE_DCT.cuda())
                loss += criterion(OUT_NOISE_DCT, noise_DCT) / (imgn_train.size()[0]*2)
                loss_DCTcomponent = criterion(OUT_NOISE_DCT, noise_DCT) / (imgn_train.size()[0]*2)


            loss.backward()
            optimizer.step()

            train_loss_log[epoch] += loss.item()
            
        train_loss_log[epoch] = train_loss_log[epoch] / len(loader_train)


        elapsed = time.time() - t
        if (opt.DCTloss_weight==1):
            print('Epoch %d: loss=%.4f, lossDCT=%.4f, elapsed time (min):%.2f' %(epoch, train_loss_log[epoch], loss_DCTcomponent.item(), elapsed/60.))
        else:
            print('Epoch %d: loss=%.4f, elapsed time (min):%.2f' %(epoch, train_loss_log[epoch], elapsed/60.))

            
        
        model_name = get_model_name(opt)
        model_dir = os.path.join('saved_models', model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, 'net_%d.pth' % (epoch)) )
        
        

if __name__ == "__main__":
    if opt.preprocess:
        if opt.color == 0:
            grayscale = True
            stride = 32
            prepare_data(data_path='training_data', patch_size=64, stride=stride, aug_times=2, grayscale=grayscale, scales_bool=True)
        else:
            stride = 64
            grayscale = False
            prepare_data(data_path='training_data', patch_size=128, stride=stride, aug_times=2, grayscale=grayscale, scales_bool=True)

    main()