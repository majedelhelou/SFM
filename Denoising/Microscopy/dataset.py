import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
# from utils import data_augmentation

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]

    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):

            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

# def prepare_data(data_path, patch_size, stride, aug_times=1, grayscale=True, scales_bool=False):
#     # train
#     print('process training data')
#     if scales_bool:
#         scales = [1, 0.9, 0.8, 0.7]
#     else:
#         scales = [1]

#     if grayscale:
#         files = glob.glob(os.path.join(data_path, 'BSD400', '*.png'))
#         files.sort()
#         train_file_name = 'train_%d.h5' % aug_times
#         if scales_bool:
#             train_file_name = 'train_%d_scales.h5' % aug_times
#     else:
#         files = glob.glob(os.path.join(data_path, 'CBSD432', '*.jpg'))
#         files.sort()
#         train_file_name = 'color_train_%d.h5' % aug_times
#         if scales_bool:
#             train_file_name = 'color_train_%d_scales.h5' % aug_times


#     h5f = h5py.File(train_file_name, 'w')
#     train_num = 0
#     for i in range(len(files)):
#         img = cv2.imread(files[i])
#         h, w, c = img.shape
#         for k in range(len(scales)):
#             Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
#             if grayscale:
#                 Img = np.expand_dims(Img[:,:,0].copy(), 0)
#             else:
#                 Img = np.rollaxis(Img, axis=2, start=0)
#             Img = np.float32(normalize(Img))

#             patches = Im2Patch(Img, win=patch_size, stride=stride)
#             print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
#             for n in range(patches.shape[3]):
#                 data = patches[:,:,:,n].copy()
#                 h5f.create_dataset(str(train_num), data=data)
#                 train_num += 1
#                 for m in range(aug_times-1):
#                     data_aug = data_augmentation(data, np.random.randint(1,8))
#                     h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
#                     train_num += 1
#     h5f.close()

#     print('training set, # samples %d\n' % train_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True, aug_times=-1, grayscale=True, scales=False):
        super(Dataset, self).__init__()
        self.train = train
        self.aug_times = aug_times
        self.grayscale = grayscale
        self.scales = scales

        if self.train:
            if self.grayscale:
                train_file_name = 'train_%d.h5' % self.aug_times
                if self.scales:
                    train_file_name = 'train_%d_scales.h5' % self.aug_times
            else:
                train_file_name = 'color_train_%d.h5' % self.aug_times
                if self.scales:
                    train_file_name = 'color_train_%d_scales.h5' % self.aug_times
            h5f = h5py.File(train_file_name, 'r')

        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            if self.grayscale:
                train_file_name = 'train_%d.h5' % self.aug_times
                if self.scales:
                    train_file_name = 'train_%d_scales.h5' % self.aug_times
            else:
                train_file_name = 'color_train_%d.h5' % self.aug_times
                if self.scales:
                    train_file_name = 'color_train_%d_scales.h5' % self.aug_times
            h5f = h5py.File(train_file_name, 'r')

        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
