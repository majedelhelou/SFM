from models.wgan_clipping import WGAN_CP
import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from utils.tensorboard_logger import Logger
from torchvision import utils
import numpy as np
import cv2
from scipy.io import savemat

def normalize_kernel(k):
    k = k.numpy()
    k = np.clip(k, 0, 1)
    k = k/sum(sum(k))
    return k  

# modify the following line to the folder of output
outputdir = 'generated_kernel/'
# modify the following line to the number of kernels that needs to be generated
num_generate = 10

if not os.path.exists(outputdir):
    os.mkdir(outputdir)

# loading model
model = WGAN_CP()
D_model_path = 'discriminator.pkl'
G_model_path = 'generator.pkl'
model.load_model(D_model_path, G_model_path)


z = Variable(torch.randn(num_generate, 100, 1, 1)).cuda()
samples = model.G(z)
samples = samples.data.cpu()
for i in range(num_generate):
    kernel = normalize_kernel(samples[i, 0])
    savemat('%s%d.mat'%(outputdir,i), {'kernel':kernel})
    
    # for plotting
    #kernel = kernel/np.max(kernel)
    #cv2.imwrite('%s%d.png'%(outputdir,i), kernel*255)