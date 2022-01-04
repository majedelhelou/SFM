# Stochastic Frequency Masking to Improve Super-Resolution and Denoising Networks

**Authors**: [Majed El Helou](https://majedelhelou.github.io/), Ruofan Zhou, and Sabine Süsstrunk

**Further readings**:
* This work inspired our [W2S](https://github.com/IVRL/w2s) (ECCVW'20), [DEU](https://github.com/IVRL/DEU) (ICIP'21), [FG-NIC](https://github.com/IVRL/FG-NIC) (SP Letters'21), and [CCID](https://github.com/IVRL/CCID) (EI'22) papers.
* We further studied the concept of restoration hallucination in [BUIFD](https://github.com/majedelhelou/BUIFD) (TIP'20), and [BIGPrior](https://github.com/majedelhelou/BIGPrior) (TIP'22). 

#### [[Paper]](https://arxiv.org/abs/2003.07119) - [[Supplementary]](https://github.com/majedelhelou/SFM/blob/master/SFM_supp.pdf) - [[Video]](https://www.youtube.com/watch?v=9ndox0p2gFg)

> **Abstract:** *Super-resolution and denoising are ill-posed yet fundamental image restoration tasks. In blind settings, the degradation kernel or the noise level are unknown. This makes restoration even more challenging, notably for learning-based methods, as they tend to overfit to the degradation seen during training.*
>
> *We present an analysis, in the frequency domain, of degradation-kernel overfitting in super-resolution and introduce a conditional learning perspective that extends to both super-resolution and denoising. Building on our formulation, we propose a stochastic frequency masking of images used in training to regularize the networks and address the overfitting problem. Our technique improves state-of-the-art methods on blind super-resolution with different synthetic kernels, real super-resolution, blind Gaussian denoising, and real-image denoising.*

**Key take-aways:** our paper explains deep SR overfitting in the frequency domain and shows that deep SR networks predict a fixed set of frequency bands, depending on training, and fail to generalize. As it is impossible to train on all possible kernels, including for methods that learn to predict those kernels, SFM simulates a spanning set for degradation kernels in the frequency domain. This improves frequency restoration and generalization. We also tie denoising to SR, in the frequency domain, and show how SFM can improve denoising performance notably on high noise levels.

## Frequency perspective 
We visualize in the frequency domain the effect of training a super-resolution network with a certain degradation kernel, and testing it on a different one. An example illustration is shown below:

<img src="figures/average_hole_plot_disk.png" width="400px"/> <img src="figures/average_hole_plot_SFM.png" width="400px"/>

The regular network fails to reconstruct a certain range of frequencies correctly (highlighted with a blue dashed circle), while the one trained with our SFM can overcome this shortcoming. The details of this experiment (whose diagram is shown below) are described in our paper, and we present further visualizations of other experiments in our supplementary material.

<p align="center">
  <img src="figures/pipeline_hole.png" width="600px"/>
</p>


## Using SFM
SFM is carried out in our paper using the DCT for transforming to the frequency domain. Other frequency transforms could be used, but for DCT a sufficient **requirement** is ```scipy.fftpack```. Below is an overview of the two modes of our SFM, explained in detail in our paper:

<p align="center">
  <img src="figures/pipeline_teaser.png" width="600px"/>
</p>

Adding a given rate of SFM into the training pipeline is a very straightforward step. Randomly select in each training batch the desired percentage of elements, and apply the mask to the selected training patches or images. The ```random_drop``` function supports the two modes of SFM described in the paper, and more functionalities are also implemented in ```utils_SFM```.

```python
import numpy as np
from utils_SFM import random_drop

DCT_DOR = 0.5 #for a 50% SFM rate (DCT dropout rate)
image_SFM = image.clone()
dct_bool = np.random.choice([1, 0], size=(image.size()[0],), p=[DCT_DOR, 1-DCT_DOR])
for img_idx in range(image.size()[0]):
    if dct_bool[img_idx] == 1:
        # random_drop settings for targeted mode (used in denoising experiments):
        image_numpy, mask = random_drop(image[img_idx,].cpu().data.numpy(), mode=2, SFM_center_radius_perc=0.85, SFM_center_sigma_perc=0.15)
        image_SFM[img_idx,] = img_numpy
image = torch.from_numpy(image_SFM).cuda()
```

## SR experiments
To reproduce our SR experiments, we make the pre-trained models and re-training code/data available under [SR](SR/). All the details can be found in the README file in that directory. Below is the SR learning pipeline with SFM:
<p align="center">
  <img src="figures/pipeline_sr.png" width="600px"/>
</p>

## Denoising experiments
To reproduce our denoising experiments, we make the pre-trained models and re-training code/data available under [Denoising](Denoising/). All the details can be found in the README file in that directory. Below is the denoising learning pipeline with SFM:
<p align="center">
  <img src="figures/pipeline_denoise.png" width="600px"/>
</p>

## Citation
```bibtex
@inproceedings{elhelou2020stochastic,
    title     = {Stochastic Frequency Masking to Improve Super-Resolution and Denoising Networks},
    author    = {El Helou, Majed and Zhou, Ruofan and S{\"u}sstrunk, Sabine},
    booktitle = {ECCV},
    year      = {2020}
}
```

