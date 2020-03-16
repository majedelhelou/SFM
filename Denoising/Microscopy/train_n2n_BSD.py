"""Training Noise2Noise model
https://github.com/NVlabs/noise2noise

Train once, test in varying imaging configurations (types) & noise levels.
Dataset: 
    training set: mixed noise levels, microscopies and cells
    test set: mixed
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from models.unet import UnetN2N, UnetN2Nv2
from utils.metrics import cal_psnr
from utils.data_loader import (load_denoising_n2n_train, 
                               load_denoising_test_mix, fluore_to_tensor)
from utils.practices import OneCycleScheduler, adjust_learning_rate, find_lr
from utils.misc import mkdirs, module_size
from utils.plot import save_samples, save_stats
import numpy as np
import argparse
import json
import random
import time
import sys
from pprint import pprint
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from torch.utils.data import DataLoader
from dataset import Dataset
from utils_SFM import random_drop
from skimage.measure.simple_metrics import compare_psnr
import glob
import os
import cv2
import pickle


def inference(test_data, model, device):
    files_source = glob.glob(os.path.join('testing_data', test_data, '*.png'))
    files_source.sort()
    
    # process data
    img_idx = 0
    std_values = list(range(10,101,10))
    psnr_results = np.zeros((len(files_source), len(std_values)))
    psnr_results2 = np.zeros((len(files_source), len(std_values)))
    
    for img_idx, f in enumerate(files_source):
        # image
        Img = cv2.imread(f)
        Img = np.float32(Img[:,:,0]) / 255.
        if Img.shape[0] % 2 == 1:
            Img = Img[1:,:]
        if Img.shape[1] % 2 == 1:
            Img = Img[:,1:]        
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        # Check dimension parity (even (h,w) for UNet):
        ISource = torch.Tensor(Img)
        
        ISource = ISource.to(device)

        for noise_idx, noise_std in enumerate(std_values):
            # create noise
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=noise_std/255.).cuda()

            # create noisy images
            INoisy = ISource + noise
            INoisy = INoisy.to(device)

            INoisy = torch.clamp(INoisy, 0., 1.)


            # feed forward then clamp image
            with torch.no_grad():
                model.eval()
                
                IDenoised = model(INoisy)
                IDenoised = torch.clamp(IDenoised, 0., 1.)
                
                Img = IDenoised.data.cpu().numpy().astype(np.float32)
                Iclean = ISource.data.cpu().numpy().astype(np.float32)
                PSNR = 0.
                for i in range(Img.shape[0]):
                    PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=1.)
                psnr_results2[img_idx, noise_idx] = PSNR
                  
    return psnr_results, psnr_results2

                    
class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Training N2N')
        self.add_argument('--exp-name', type=str, default='n2n', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')        
        self.add_argument('--post', action='store_true', default=False, help='post proc mode')
        self.add_argument('--debug', action='store_true', default=False, help='verbose stdout')
        self.add_argument('--net', type=str, default='unet', choices=['unet', 'unetv2'])
        # data
        self.add_argument('--data-root', type=str, default="./dataset", help='directory to dataset root')
        self.add_argument('--imsize', type=int, default=256)
        self.add_argument('--in-channels', type=int, default=1)
        self.add_argument('--out-channels', type=int, default=1)
        self.add_argument('--transform', type=str, default='center_crop', choices=['four_crop', 'center_crop'])
        self.add_argument('--noise-levels-train', type=list, default=[1, 2, 4, 8, 16])
        self.add_argument('--noise-levels-test', type=list, default=[1])
        self.add_argument('--test-group', type=int, default=19)
        self.add_argument('--captures', type=int, default=50, help='how many captures in each group to load')
        # training
        self.add_argument('--epochs', type=int, default=400, help='number of iterations to train')
        self.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
        self.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.add_argument('--wd', type=float, default=0., help="weight decay")
        self.add_argument('--test-batch-size', type=int, default=2, help='input batch size for testing')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        self.add_argument('--cuda', type=int, default=0, help='cuda number')
        # logging
        self.add_argument('--ckpt-freq', type=int, default=10, help='how many epochs to wait before saving model')
        self.add_argument('--print-freq', type=int, default=282, help='how many minibatches to wait before printing training status')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-epochs', type=int, default=5, help='how many epochs to wait before plotting test output')
        self.add_argument('--cmap', type=str, default='inferno', help='attach notes to the run dir')
        # Blind + SFM
        self.add_argument('--noise_max', type=int, default=55, help='max noise level seen during blind denoising training')
        self.add_argument('--DCT_DOR', type=float, default=0, help='DCT Dropout Rate, if 0 no DCT dropout')
        
    def parse(self):
        args = self.parse_args()
        date = '{}'.format(time.strftime('%b_%d'))
        args.run_dir = args.exp_dir + '/' + args.exp_name + '/' + date \
            + f'/{args.net}_noise_train{args.noise_levels_train}_'\
            f'test{args.noise_levels_test}_{args.transform}_'\
            f'epochs{args.epochs}_bs{args.batch_size}_lr{args.lr}'\
            f'DCTDOR{args.DCT_DOR}'
        args.ckpt_dir = args.run_dir + '/checkpoints'

        if not args.post:
            mkdirs([args.run_dir, args.ckpt_dir])

        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark=True

        print('Arguments:')
        pprint(vars(args))

        if not args.post:
            with open(args.run_dir + "/args.txt", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args

args = Parser().parse()
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])
if args.net == 'unet':
    model = UnetN2N(args.in_channels, args.out_channels).to(device)
elif args.net == 'unetv2':
    model = UnetN2Nv2(args.in_channels, args.out_channels).to(device)

if args.debug:
    print(model)
    print(model.model_size)


dataset_train = Dataset(train=True, aug_times=2, grayscale=True, scales=True)
train_loader = DataLoader(dataset=dataset_train, num_workers=4, batch_size=args.batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
    weight_decay=args.wd, betas=[0.9, 0.99])
lr = args.lr

multiplier = 4 if args.transform == 'four_crop' else 1
n_train_samples = len(train_loader.dataset) * multiplier
pixels_per_sample = train_loader.dataset[0][0].numel()
n_train_pixels = n_train_samples * pixels_per_sample

np.random.seed(0)
torch.manual_seed(0)

logger = {}
logger['psnr_train'] = []
logger['rmse_train'] = []
logger['psnr_test'] = []
logger['rmse_test'] = []

total_steps = args.epochs * len(train_loader)
print('Start training........................................................')
try:
    tic = time.time()
    iters = 0
    for epoch in range(1, args.epochs + 1):
        model.train()

        psnr, mse = 0., 0.
        for batch_idx, (clean) in enumerate(train_loader):
            iters += 1
            
            noise = torch.zeros(clean.size())
            noise2 = torch.zeros(clean.size())
            stdN = np.random.uniform(0, args.noise_max, size=noise.size()[0])
            sizeN = noise[0,:,:,:].size()
            for n in range(noise.size()[0]):
                noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
                noise2[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            
            noisy_input = clean + noise
            noisy_target = clean + noise2
            
            noisy_input, noisy_target, clean = noisy_input.to(device), \
                noisy_target.to(device), clean.to(device)
            
            
            # DCT SFM
            if args.DCT_DOR > 0:
                noisy_input_SFM = np.zeros(noisy_input.size(),dtype='float32')
                dct_bool = np.random.choice([1, 0], size=(noisy_input.size()[0],), p=[args.DCT_DOR, 1-args.DCT_DOR])
                for img_idx in range(noisy_input.size()[0]):
                    if dct_bool[img_idx] == 1:
                        
                        img_numpy, mask = random_drop(noisy_input[img_idx,:,:,:].cpu().data.numpy(), mode=2, SFM_center_radius_perc=0.85, SFM_center_sigma_perc=0.15)
                        noisy_input_SFM[img_idx,0,:,:] = img_numpy
                noisy_input = torch.from_numpy(noisy_input_SFM).cuda()
                
                
            model.zero_grad()
            denoised = model(noisy_input)
            loss = F.mse_loss(denoised, noisy_target, reduction='sum')
            loss.backward()


            lr = lr * 10**(-np.floor(epoch/30))
            adjust_learning_rate(optimizer, lr)

            optimizer.step()

            mse += loss.item()
            with torch.no_grad():
                clean255 = clean.add(0.5).mul(255).clamp(0, 255)
                denoised255 = denoised.detach().add(0.5).mul(255).clamp(0, 255)
                mse = F.mse_loss(denoised255, clean255, reduction='none').view(clean.shape[0], -1).mean(1)
                mse = mse.sum().item()
                psnr += 10 * np.log10(255 ** 2 / mse)

            if iters % args.print_freq == 0:
                print(f'[{batch_idx+1}|{len(train_loader)}]'\
                    f'[{epoch}|{args.epochs}] training PSNR: '\
                    f'{(psnr / (batch_idx+1) / args.batch_size / multiplier):.6f}')
        print(f'Epoch {epoch}, lr {lr}')
         
        psnr = psnr / n_train_samples
        rmse = np.sqrt(mse / n_train_pixels)
        
        if epoch % args.log_freq == 0:
            logger['psnr_train'].append(psnr)
            logger['rmse_train'].append(rmse)
        print("Epoch {} training PSNR: {:.6f}, RMSE: {:.6f}".format(epoch, psnr, rmse))

        # save model
        if epoch % args.ckpt_freq == 0:
            torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))


        # test ------------------------------
        if epoch % 1 == 0:
            results_dir = args.run_dir + '/checkpoints'

            # BSD68
            psnr_results, psnr_results2  = inference('BSD68', model, device)
            results_file =  os.path.join(results_dir, "_data%d.pkl" %epoch)
            with open(results_file, "wb") as f:
                pickle.dump(psnr_results2, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            psnr_results, psnr_results2  = inference('Set12', model, device)
            results_file =  os.path.join(results_dir, "_Set12data%d.pkl" %epoch)
            with open(results_file, "wb") as f:
                pickle.dump(psnr_results2, f, protocol=pickle.HIGHEST_PROTOCOL)

                
    tic2 = time.time()
    print("Finished training {} epochs using {} seconds"
        .format(args.epochs, tic2 - tic))

    x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)

    args.training_time = tic2 - tic
    args.n_params, args.n_layers = module_size(model)
    with open(args.run_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)

except KeyboardInterrupt:
    print('Keyboard Interrupt captured...Saving models & training logs')
    tic2 = time.time()
    torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))
    x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
    # plot the rmse, r2-score curve and save them in txt
    save_stats(args.train_dir, logger, x_axis, 'psnr_train', 'psnr_test', 
        'rmse_train', 'rmse_test')

    args.training_time = tic2 - tic
    args.n_params, args.n_layers = module_size(model)
    with open(args.run_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)
