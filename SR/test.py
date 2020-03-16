import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
#from model import RCAN, SRFBN, RRDB, SAN, RDN, EDSR
import skimage

from model.RRDB import RRDBNet
from model.SRFBN import SRFBN
from model.SAN import SAN
from model.RDN import RDN
from model.EDSR import EDSR
from model.RCAN import RCAN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--epoch", type=int, default=48, help="Number of training epochs")
parser.add_argument("--net", type=str, default="RCAN", help="RCAN, ESRGAN, RRDB, KMSR")
parser.add_argument("--test_path", type=str, default='test_imgs/DIV2K/', help="Directory where test images are stored")
parser.add_argument("--gt_path", type=str, default='test_imgs/DIV2K/', help="Directory where GT images are stored")
parser.add_argument("--scale", type=int, default=4, help='scaling factor of SR')
opt = parser.parse_args()


def inference(model, test_path, gt_path, results_dir):
    files_input = glob.glob(test_path+'/*.png')
    files_input.sort()
    files_gt = glob.glob((gt_path+'/*.png'))
    files_gt.sort()
    
    psnr_results = np.zeros(len(files_input))
    ssim_results = np.zeros(len(files_input))
    
    for idx in range(len(files_input)):
        lr_img = cv2.imread(files_input[idx], cv2.IMREAD_GRAYSCALE)
        (w,h) = np.shape(lr_img)
        img_ans = np.zeros((w*2,h*2))
        x = 0
        y = 0
        while (x < w):
            y = 0
            while (y < h):
                tmp_patch = lr_img[x:x+128,y:y+128] / 255
                tmp_patch = np.expand_dims(tmp_patch, 0)
                tmp_patch = torch.tensor([tmp_patch]).float()
                sr_patch = model(tmp_patch)
                if isinstance(sr_patch, list):
                    sr_patch = sr_patch[-1]
                sr_patch = sr_patch.cpu().data.numpy()
                #print(sr_patch)
                sr_patch = np.clip(sr_patch, 0, 1)
                sr_patch = sr_patch[0][0]

                img_ans[x*2+64:x*2+256,y*2+64:y*2+256] = sr_patch[64:,64:]
                if (x < 64):
                    img_ans[x*2:x*2+64,y*2+64:y*2+256] = sr_patch[:64,64:]
                if (y < 64):
                    img_ans[x*2+64:x*2+256,y*2:y*2+64] = sr_patch[64:,:64]
                if (x < 64 and y < 64):
                    img_ans[x*2:x*2+64,y*2:y*2+64] = sr_patch[:64,:64]
                y = y + 64
            x = x + 64
        
        img_gt = cv2.imread(files_gt[idx], cv2.IMREAD_GRAYSCALE)
        #print(skimage.measure.compare_psnr(img_ans, img_gt/255))
        psnr_results[idx] = skimage.measure.compare_psnr(img_ans, img_gt/255)
        ssim_results[idx] = skimage.measure.compare_ssim(img_ans, img_gt/255)
        cv2.imwrite(results_dir + '/' + os.path.basename(files_input[idx]), (255*img_ans).astype('uint8'))
    return psnr_results, ssim_results



def main():
    
    model_name = 'checkpoints/' + opt.net
    
    if (opt.net == 'RCAN'):
    model = RCAN(scale=scale)
    elif (opt.net == 'RRDB'):
        model = RRDBNet(scale=scale)
    elif (opt.net == 'KMSR'):
        model = ResBlockNet()
    elif (opt.net == 'ESRGAN'):
        model = RRDBNet(scale=scale)
    else:
        raise NotImplemented('Network model not implemented.')

    model = nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'epoch_%d.pth' % (opt.epoch)))['model'].state_dict())
    model.eval()

    if opt.denoiser == 'None':
        test_dir = os.path.join(opt.test_path, opt.avg)
        result_dir = os.path.join('results/', opt.net, opt.avg)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:    
        test_dir = os.path.join(opt.test_path, opt.denoiser)
        result_dir = os.path.join(opt.test_path, opt.net, opt.denoiser)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    
    print('Testing with model %s at epoch %d, with %s' %(model_name, opt.epoch, test_dir))

    if opt.denoiser == 'None':
        psnr_results, ssim_results = inference(model, test_dir, opt.gt_path, result_dir)
        np.save(os.path.join(result_dir, 'PSNR'), psnr_results)
        np.save(os.path.join(result_dir, 'SSIM'), ssim_results)
        return
    for avg_img_dirs in glob.glob('%s/*'%(test_dir)):
        test_dir = avg_img_dirs
        results_dir = os.path.join(result_dir, os.path.basename(test_dir))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        psnr_results, ssim_results = inference(model, test_dir, opt.gt_path, results_dir)
        print( 'Average %s PSNR: %.2fdB' %(opt.test_path, np.mean(psnr_results)) ) 
        np.save( os.path.join(results_dir, 'PSNR'), psnr_results )
        np.save( os.path.join(results_dir, 'SSIM'), ssim_results )

if __name__ == "__main__":
    main()
