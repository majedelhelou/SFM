"""
SFM: Stochastic Frequency Masking
"""
import random
from scipy.fftpack import dct, idct
import numpy as np

def fully_random_drop_mask(w=256,h=256,radius=0,p=0.5):
    mask_random = np.random.rand(w,h)
    mask = np.ones((w,h))
    mask[mask_random > p] = 0
    
    return mask


def circular_random_drop_mask(w=256, h=256, SFM_center_radius_perc=-1, SFM_center_sigma_perc=0.05):
    ''' 
    (w,h) are the dimensions of the mask
    
    IF (SFM_center_radius_perc=-1)
        the masked regions are selected randomly in circular shape, with the maximum at "radius"
        when "radius" is 0, it is set to the max default value
    
    ELSE
        the masked regions are always centered at "SFM_center_radius_perc*radius", and stretch inwards and 
        outwards with a Gaussian probability, with sigma=SFM_center_sigma_perc*radius
    '''
    
    radius = np.sqrt(w*w+h*h)
    SFM_center_sigma = SFM_center_sigma_perc * radius
    SFM_center_radius = SFM_center_radius_perc * radius
    
    X, Y = np.meshgrid(np.linspace(0,h-1,h), np.linspace(0,w-1,w))
    D = np.sqrt(X*X+Y*Y)
    
    #random SFM (SFM_center_radius 0) vs SFM around a center of given distance
    if SFM_center_radius_perc == -1:    
        a1 = random.random()*radius
        a2 = random.random()*radius
        if (a1 > a2):
            tmp = a2;a2 = a1;a1 = tmp
        mask = np.ones((w,h))
        mask[(D>a1)&(D<a2)] = 0
        
    else:
        if SFM_center_radius > radius or SFM_center_radius < 0:
            raise Exception('SFM_center_radius out of bounds.')
        
        a1 = random.gauss(0, SFM_center_sigma)
        a2 = random.gauss(0, SFM_center_sigma)
        
        a1 = abs(a1)
        a2 = abs(a2)
        
        a1 = SFM_center_radius - a1
        a2 = SFM_center_radius + a2
        
        mask = np.ones((w,h))
        mask[(D>a1)&(D<a2)] = 0
        
    return mask


def random_drop(img, mode=1, SFM_center_radius_perc=-1, SFM_center_sigma_perc=0.05):
    ''' mode=0:fully random drop, mode=1: circular random drop, mode=2 sweeping mode
        
        **sweeping mode**:
            SFM_center_radius_perc: determines the center of the band to be erased
                                    it is a percentage of the max radius
            SFM_center_sigma_perc:  determines the sigma for the width of the band
                                    sigma=radius*SFM_center_sigma_perc
    '''
    
    (c,w,h) = np.shape(img)
    if mode == 0:
        mask = fully_random_drop_mask(w,h)
    if mode == 1:
        mask = circular_random_drop_mask(w,h)
    if mode == 2:
        mask = circular_random_drop_mask(w, h, SFM_center_radius_perc, SFM_center_sigma_perc)
    
    if c == 3:
        img0_dct = dct(dct(img[0,:,:], axis=0, norm='ortho'), axis=1, norm='ortho')
        img1_dct = dct(dct(img[1,:,:], axis=0, norm='ortho'), axis=1, norm='ortho')
        img2_dct = dct(dct(img[2,:,:], axis=0, norm='ortho'), axis=1, norm='ortho')
        img0_dct = img0_dct*mask
        img1_dct = img1_dct*mask
        img2_dct = img2_dct*mask
        img[0,:,:]= idct(idct(img0_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        img[1,:,:]= idct(idct(img1_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        img[2,:,:]= idct(idct(img2_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    elif c == 1:
        img_dct = dct(dct(img[0,:,:], axis=0, norm='ortho'), axis=1, norm='ortho')
        img_dct = img_dct*mask
        img[0,:,:]= idct(idct(img_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        
    return (img, mask)






def get_mask_low_high(w=256, h=256, radius_perc=-1, mask_mode=-1):
    ''' 
    (w,h) are the dimensions of the mask
    if mask_mode==1 low frequencies are cut off
    if mask_mode==2 high frequencies are cut off
    
    returns a binary mask of low or of high frequencies, cut-off at radius_perc*radius
    '''
        
    if radius_perc < 0:
        raise Exception('radius_perc must be positive.')
    
    radius = np.sqrt(w*w+h*h)
    center_radius = radius_perc * radius
    
    X, Y = np.meshgrid(np.linspace(0,h-1,h), np.linspace(0,w-1,w))
    D = np.sqrt(X*X+Y*Y)
    
    if mask_mode == 1:
        a1 = 0
        a2 = center_radius
    elif mask_mode == 2:
        a1 = center_radius
        a2 = radius
    else:
        raise Exception('mask_mode must be 1 or 2.')
    
    mask = np.ones((w,h))
    mask[(D>=a1)&(D<=a2)] = 0
        
    return mask