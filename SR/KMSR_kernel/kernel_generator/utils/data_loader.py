import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch.utils.data as data
import torch
from torchvision import transforms
from functools import partial
import numpy as np
from imageio import imread
import glob
from scipy.io import loadmat

class FolderDataset(data.Dataset):
    def __init__(self, dataset='x2'):
        super(FolderDataset, self).__init__()
        
        base = dataset
        
        self.mat_files = sorted(glob.glob(base + '*.mat'))
        
    def __getitem__(self, index):
        mat = loadmat(self.mat_files[index])
        x = np.array([mat['kernel']])
        #x = np.swapaxes(x, 2, 0)
        #print(np.shape(x))
        
        return torch.from_numpy(x).float()
        
    def __len__(self):
        return len(self.mat_files)
