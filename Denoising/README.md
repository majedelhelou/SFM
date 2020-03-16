# Denoising experiments

We provide the pre-trained models and their results, and the training code/data to reproduce the learning methods. This covers our experiments on [AWGN](https://github.com/majedelhelou/SFM/Denoising#AWGN) denoising and [real microscopy Poisson-Gaussian](https://github.com/majedelhelou/SFM/Denoising#real-microscopy) denoising. 

## AWGN
### Pre-trained
The pretrained network models (with and without SFM training) are in:

- **DnCNN-B**: [regular](https://github.com/majedelhelou/SFM/tree/master/Denoising/saved_models/gray_DnCNN_55_SFMm0_0.00_Noise0_GT0)/[SFM](https://github.com/majedelhelou/SFM/tree/master/Denoising/saved_models/gray_DnCNN_55_SFMm2_0.50_Noise0_GT0_rad_0.85_radsig_0.15)

- **MemNet**: [regular](https://github.com/majedelhelou/SFM/tree/master/Denoising/saved_models/gray_MemNet_55_SFMm0_0.00_Noise0_GT0)/[SFM](https://github.com/majedelhelou/SFM/tree/master/Denoising/saved_models/gray_MemNet_55_SFMm2_0.50_Noise0_GT0_rad_0.85_radsig_0.15)

- **RIDNet**: [regular](https://github.com/majedelhelou/SFM/tree/master/Denoising/saved_models/gray_RIDNet_55_SFMm0_0.00_Noise0_GT0)/[SFM](https://github.com/majedelhelou/SFM/tree/master/Denoising/saved_models/gray_RIDNet_55_SFMm2_0.50_Noise0_GT0_rad_0.85_radsig_0.15)

- **N2N**: [regular](https://github.com/majedelhelou/SFM/tree/master/Denoising/Microscopy/experiments/n2n/Nov_04/unet_noise_train%5B1%2C%202%2C%204%2C%208%2C%2016%5D_test%5B1%5D_center_crop_epochs51_bs64_lr0.0001DCTDOR0)/[SFM](https://github.com/majedelhelou/SFM/tree/master/Denoising/Microscopy/experiments/n2n/Nov_04/unet_noise_train%5B1%2C%202%2C%204%2C%208%2C%2016%5D_test%5B1%5D_center_crop_epochs51_bs64_lr0.0001DCTDOR0.5)

- **N3Net**: [regular](https://github.com/majedelhelou/SFM/tree/master/Denoising/N3Net/results_gaussian_denoising/0002-/checkpoint)/[SFM](https://github.com/majedelhelou/SFM/tree/master/Denoising/N3Net/results_gaussian_denoising/0003-/checkpoint)

PSNR results of DnCNN-B, MemNet and RIDNet are in [PSNR_results](https://github.com/majedelhelou/SFM/tree/master/Denoising/PSNR_results), those of N2N are in this [notebook](https://github.com/majedelhelou/SFM/blob/master/Denoising/Microscopy/view_results_BSD.ipynb) and those of N3Net are in this [notebook](https://github.com/majedelhelou/SFM/blob/master/Denoising/N3Net/view_results.ipynb).

### Re-training
To reproduce the regular training of:
- **DnCNN-B**/**MemNet**/**RIDNet** (net_mode R/M/D respectively):

```python train.py --net_mode R```

Add ```--preprocess True``` if it is the first time you train, to generate the dataset.

And to train with SFM (50% rate, targeted mode at 0.85 radius, with sigma 0.15):

```python train.py --net_mode R --DCT_DOR 0.5 --SFM_mode 2 --SFM_rad_perc 0.85 --SFM_sigma_perc 0.15```

- **N2N**

Copy the generated patch dataset ```train_2_scales.h5``` to the ```Microscopy``` directory and from there you can run:

```python train_n2n_BSD.py```

Add ```--DCT_DOR  0.5``` to train with SFM, it is set by default to the same center radius and sigma values (0.85/0.15) for the targeted mode of our SFM.

- **N3Net** 

N3Net requires a specific setup to run, please refer to the original [N3Net repository](https://github.com/visinf/n3net). Follow the installation guidelines of N3Net and **replace** the original ```experiment.py``` and ```main.py``` files of the installation by the same 2 files that we provide under the ```N3Net``` directory to support SFM. Similar to our N2N, you can pass the DCT_DOR argument to the ```main.py``` file.



## Real microscopy
Inside the directory ```Microscopy```, follow the [FMD instructions](https://github.com/bmmi/denoising-fluorescence#fmd-dataset) to install the microscopy dataset.

### Classical methods
The ```matlab``` directory contains the different classical methods used for reference. The code is adapted from the [FMD repository](https://github.com/bmmi/denoising-fluorescence).

### Pre-trained
The pretrained network models (with and without SFM training) are in:
- **N2S** [regular](https://github.com/majedelhelou/SFM/tree/master/Denoising/Microscopy/experiments/n2s/Oct_18/unet_noise_train%5B1%2C%202%2C%204%2C%208%2C%2016%5D_test%5B1%5D_four_crop_epochs100_bs4_lr1e-05SFM0/checkpoints)/[SFM](https://github.com/majedelhelou/SFM/tree/master/Denoising/Microscopy/experiments/n2s/Oct_18/unet_noise_train%5B1%2C%202%2C%204%2C%208%2C%2016%5D_test%5B1%5D_four_crop_epochs100_bs4_lr1e-05SFM0.5/checkpoints)

- **N2N** [regular](https://github.com/majedelhelou/SFM/tree/master/Denoising/Microscopy/experiments/n2n/Oct_17/unet_noise_train%5B1%2C%202%2C%204%2C%208%2C%2016%5D_test%5B1%5D_four_crop_epochs400_bs4_lr0.0001SFM0/checkpoints)/[SFM](https://github.com/majedelhelou/SFM/tree/master/Denoising/Microscopy/experiments/n2n/Oct_17/unet_noise_train%5B1%2C%202%2C%204%2C%208%2C%2016%5D_test%5B1%5D_four_crop_epochs400_bs4_lr0.0001SFM0.5/checkpoints)

PSNR results are in JSON format in ```benchmark_gpu/results_gpu.txt``` in the directories of each pre-trained model, and can be visualized with this [notebook](https://github.com/majedelhelou/SFM/blob/master/Denoising/Microscopy/view_results_JSON.ipynb).


### Re-training
- **N2S**

```python train_n2s.py```

We implement the masking procedure in ```mask.py``` that corresponds to the description of the original N2S paper (it performs significantly better than the procedures provided -at the time of writing- in the N2S repository).
Similar to training N2N for the AWGN noise removal, to train with SFM add ```--DCT_DOR  0.5```, and you get the default setup for the targeted SFM mode.

- **N2N**

```python train_n2n.py```

Similar to training N2N for the AWGN noise removal, to train with SFM add ```--DCT_DOR  0.5```, and you get the default setup for the targeted SFM mode.

