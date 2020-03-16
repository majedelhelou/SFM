import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN_RL, BUIFD, MemNet, get_model_name
from models import RIDNET as RIDNet
from utils import *
from utils_SFM import *
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--net_mode", type=str, default='R', help='Model chosen for testing: R (DnCNN), M (MemNet) or D (RIDNet)')
parser.add_argument("--noise_max", type=int, default=55, help="Training noise level range")
parser.add_argument("--DCT_DOR", type=float, default=0, help="DCT Dropout Rate, if 0 no DCT dropout")
parser.add_argument("--SFM_mode", type=int, default=0, help="0 fully random, 1 circular, 2 sweeping")
parser.add_argument("--SFM_rad_perc", type=float, default=0.5, help="IN MODE 2: the central band for SFM is SFM_rad_perc*max_radius")
parser.add_argument("--SFM_sigma_perc", type=float, default=0.05, help="IN MODE 2: the stddev around the central band for SFM")
parser.add_argument("--SFM_noise", type=int, default=0, help="0: SFM input before adding noise, 1 after")
parser.add_argument("--SFM_GT", type=int, default=0, help="0: don't SFM the GT, 1 SFM GT with input's mask")
parser.add_argument("--mask_train_noise", type=int, default=0, help='if 0: nothing; if 1: high frequency noise only; if 2: low frequency noise only; if 3: Brownian noise')
parser.add_argument("--mask_test_noise", type=int, default=0, help='if 0: white; if 1: high frequency test noise; if 2: low frequency test noise; if 3: Brownian noise')
parser.add_argument("--DCTloss_weight", type=int, default=0, help='if 0: nothing; if 1: adds auxiliary DCT-reconstruction loss')
parser.add_argument("--varying_noise", type=bool, default=False, help="Set to True if varying noise is used")
parser.add_argument("--color_mode", type=str, default='gray', help='Grayscale (gray) or color (color) model')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of res blocks used in the model")

opt = parser.parse_args()

def normalize(data):
    return data/255.

def create_varying_noise(image_size, noise_std_min, noise_std_max):
    ''' outputs a noise image of size image_size, with varying noise levels, ranging from noise_std_min to noise_std_max
    the noise level increases linearly with the number of rows in the image '''
    noise = torch.FloatTensor(image_size).normal_(mean=0, std=0).cuda()

    row_size = torch.Size([image_size[0], image_size[1], image_size[2]])
    for row in range(image_size[3]):
        std_value = noise_std_min + (noise_std_max-noise_std_min) * (row/(image_size[3]*1.0-1))
        noise[:,:,:,row] = torch.FloatTensor(row_size).normal_(mean=0, std=std_value/255.).cuda()

    return noise


def inference(test_data, model, varying_noise=False, color_mode='gray'):
    files_source = glob.glob(os.path.join('testing_data', test_data, '*.png'))

    files_source.sort()
    # process data
    img_idx = 0
    std_values = list(range(5,101,5))
    psnr_results = np.zeros((len(files_source), len(std_values)))
    DCT_MSE_results = np.zeros((len(files_source), len(std_values), 2)) # 2: 0 for low DCT band, 1 for high DCT band
    
    for f in files_source:
        # image
        if color_mode == 'color':
            Img = cv2.imread(f)
            Img = normalize(np.float32(Img[:,:,:]))
            Img = np.rollaxis(Img, axis=2, start=0)
            Img = np.expand_dims(Img, 0)
            ISource = torch.Tensor(Img)
            ISource = Variable(ISource.cuda())
        else:
            Img = cv2.imread(f)
            Img = normalize(np.float32(Img[:,:,0]))
            Img = np.expand_dims(Img, 0)
            Img = np.expand_dims(Img, 1)
            ISource = torch.Tensor(Img)
            ISource = Variable(ISource.cuda())

        for noise_idx, noise_std in enumerate(std_values):
            # create noise
            if varying_noise:
                if noise_std not in [15,25,40,55,65]:
                    continue
                noise = create_varying_noise(ISource.size(), noise_std - 10, noise_std + 10)
            else:
                noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=noise_std/255.).cuda()

                
            # Mask the noise to use white/high/low testing noise
            if opt.mask_test_noise in([1,2]):
                noise_mask = get_mask_low_high(w=ISource.size()[2], h=ISource.size()[3], radius_perc=0.5, mask_mode=opt.mask_train_noise)
                
                noise_dct = dct(dct(noise[0,0,:,:].data.cpu().numpy(), axis=0, norm='ortho'), axis=1, norm='ortho')
                noise_dct = noise_dct * noise_mask
                noise_numpy = idct(idct(noise_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
                noise[0,0,:,:] = torch.from_numpy(noise_numpy)
            elif opt.mask_test_noise == 3: #Brownian noise
                noise_numpy = gaussian_filter(noise[0,0,:,:].data.cpu().numpy(), sigma=3)
                noise[0,0,:,:] = torch.from_numpy(noise_numpy)    


            # create noisy images
            INoisy = ISource + noise
            INoisy = Variable(INoisy.cuda())
            INoisy = torch.clamp(INoisy, 0., 1.)


            # feed forward then clamp image
            with torch.no_grad():
                NoiseNetwork = model(INoisy)
                INetwork = torch.clamp(NoiseNetwork, 0., 1.)
                psnr_results[img_idx, noise_idx] = batch_PSNR(INetwork, ISource, 1.)
        
                # Get low and high frequency DCT to compute MSE separately on each
                sizeN = INetwork.size()
                HP_mask = get_mask_low_high(w=sizeN[2], h=sizeN[3], radius_perc=0.5, mask_mode=1)
                LP_mask = get_mask_low_high(w=sizeN[2], h=sizeN[3], radius_perc=0.5, mask_mode=2)
                
                INetwork_numpy = INetwork.data.cpu().numpy()[0,0,:]
                INetwork_numpy_DCT = dct(dct(INetwork_numpy, axis=0, norm='ortho'), axis=1, norm='ortho')
                ISource_numpy = ISource.data.cpu().numpy()[0,0,:]
                ISource_numpy_DCT = dct(dct(ISource_numpy, axis=0, norm='ortho'), axis=1, norm='ortho')
                    
                INetwork_DCT_L = LP_mask * INetwork_numpy_DCT
                INetwork_DCT_H = HP_mask * INetwork_numpy_DCT
                ISource_DCT_L = LP_mask * ISource_numpy_DCT
                ISource_DCT_H = HP_mask * ISource_numpy_DCT
                                            
                DCT_MSE_results[img_idx, noise_idx, 0] = np.mean( (INetwork_DCT_L - ISource_DCT_L)**2 )
                DCT_MSE_results[img_idx, noise_idx, 1] = np.mean( (INetwork_DCT_H - ISource_DCT_H)**2 )


        img_idx += 1
        
    return psnr_results, DCT_MSE_results



def main():
    
    opt.color = int(opt.color_mode == 'color')
    model_name = get_model_name(opt)
    log_dir = os.path.join('PSNR_Results', model_name)
    model_dir = os.path.join('saved_models', model_name)

    
    print('Testing with model %s, with %s' %(model_name, 'varying noise...' if opt.varying_noise else 'uniform noise...'))

    
    # Build model:
    num_of_layers = opt.num_of_layers
    model_channels = (3 if opt.color_mode == 'color' else 1)
    if opt.net_mode == 'R':
        net = DnCNN_RL(channels=model_channels, num_of_layers=num_of_layers)
    elif opt.net_mode == 'M':
        net = MemNet(in_channels=model_channels, channels=20, num_memblock=6, num_resblock=4)
    elif opt.net_mode == 'D':
        net = RIDNet(in_channels=model_channels)
    else:
        raise NotImplemented('Supported networks: R (DnCNN), M (MemNet), D (RIDNet) only')

    # Load model:
    model = nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth' )))
    model.eval()


    test_data = 'BSD68'
    psnr_results, DCT_MSE_results = inference(test_data, model, opt.varying_noise, opt.color_mode)

    
    string = ''
    for std_idx, std in enumerate(range(5,101,5)):
        string += '& %.2f ' %(np.mean(psnr_results[:,std_idx]))
    print( string )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    psnr_results_df = pd.DataFrame(psnr_results)
    psnr_results_df.to_csv(os.path.join( log_dir, 'PSNR_' + str(opt.mask_test_noise) ) + '.csv', index=False)
    DCT_MSE_results_df = pd.DataFrame(DCT_MSE_results[:,:,0])
    DCT_MSE_results_df.to_csv(os.path.join( log_dir, 'LDCT_MSE_' + str(opt.mask_test_noise) ) + '.csv', index=False)
    DCT_MSE_results_df = pd.DataFrame(DCT_MSE_results[:,:,1])
    DCT_MSE_results_df.to_csv(os.path.join( log_dir, 'HDCT_MSE_' + str(opt.mask_test_noise) ) + '.csv', index=False)


if __name__ == "__main__":
    main()