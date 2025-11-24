import math
from torch import gt
import torch.nn.functional as f
import os
from PIL import Image
from torchvision import transforms
import random

def PSNR(truth, pred):
    mse = f.mse_loss(pred,truth,reduction='mean').item()
    if mse == 0: # same image
        return 100
    return 20*math.log10(1.0/math.sqrt(mse))
    


def compute_PSNR(truth_folder, pred_folder, sample_size):
    to_tensor = transforms.ToTensor()

    total = 0.0
    gt_files = sorted(os.listdir(truth_folder))
    gt_files = random.sample(gt_files, sample_size)

    for fname in gt_files:
        gt_path = os.path.join(truth_folder, fname)
        pred_path = os.path.join(pred_folder, fname)


        gt_img = Image.open(gt_path).convert('RGB')
        pred_img = Image.open(pred_path).convert('RGB')
        gt_img = gt_img.resize(pred_img.size, Image.BILINEAR)

        gt_tensor = to_tensor(gt_img).unsqueeze(0)
        pred_tensor = to_tensor(pred_img).unsqueeze(0)

        psnr = PSNR(gt_tensor, pred_tensor)
        total+=psnr


    return total / sample_size