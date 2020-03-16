import os
import glob
import shutil

# modify the following line to the folder of the estimated kernels
base = '../../kernel_estimation/x2results/'
folders = ['blackberry_x2/', 'sony_x2/']
# modify the following line to the folder of generated 
outputdir = 'x2/'

if not os.path.exists(outputdir):
    os.mkdir(outputdir)

count = 0
for folder in folders:
    imgs = glob.glob(base + folder + '/*.mat')
    xfor mat in imgs:
        shutil.copy2(mat, outputdir + str(count) + '.mat')
        count = count + 1
