# Super-resolution experiments

We provide the pre-trained models, and the training code/data to reproduce the learning methods. This covers our experiments on synthetic super-resolution and real-world super-resolution.

## Synthetic SR
### Pre-trained models
Due to size issue and for anonymity, the pre-trained models will be uploaded to Google Drive later


### Re-training
To reproduce the regular training of:
- **RCAN**/**RRDB**/**ESRGAN** (net RCAN/RRDB/ESRGAN respectively):

```python train.py --net RCAN```

And to train with SFM (50% rate, central mode):

```python train.py --net_mode RCAN --SFM 1```

- **KMSR**

To after generate kernels in folder KMSR_kernel, run:

```python train.py --net KMSR```

Add ```--SFM 1``` to train with SFM for the central mode of our SFM.

- **IKC** 

We use the original [IKC protocal](https://github.com/yuanjunchai/IKC). Similar to our other training codes, you can pass the SFM argument to the ```train_IKC.py``` file.


## Real-world super-resolution
### Pre-trained
Due to size issue and for anonymity, the pre-trained models will be uploaded to Google Drive later

### Re-training
The same as the training process for synthetic dataset except for replacing the training data.
