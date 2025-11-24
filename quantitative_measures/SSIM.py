import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as f
import random

def SSIM(truth, pred):
    pred_gray = pred.mean(dim=1, keepdim=True)
    target_gray = truth.mean(dim=1, keepdim=True)


    mu_pred = f.avg_pool2d(pred_gray, 3, 1, 0)
    mu_target = f.avg_pool2d(target_gray, 3, 1, 0)


    sigma_pred = f.avg_pool2d(pred_gray * pred_gray, 3, 1, 0) - mu_pred ** 2
    sigma_target = f.avg_pool2d(target_gray * target_gray, 3, 1, 0) - mu_target ** 2
    sigma_cross = f.avg_pool2d(pred_gray * target_gray, 3, 1, 0) - mu_pred * mu_target


    ssim_n = (2 * mu_pred * mu_target + (0.01**2)) * (2 * sigma_cross + (0.03**2))
    ssim_d = (mu_pred ** 2 + mu_target ** 2 + (0.01**2)) * (sigma_pred + sigma_target + (0.03**2))


    ssim_map = ssim_n / ssim_d
    return ssim_map.mean().item()

def compute_SSIM(truth_folder, pred_folder, sample_size):
    to_tensor = transforms.ToTensor()
    total = 0.0
    gt_files = sorted(os.listdir(truth_folder))
    gt_files = random.sample(gt_files, sample_size)

    for fname in gt_files:
        gt_path = os.path.join(truth_folder, fname)
        pred_path = os.path.join(pred_folder, fname)

        gt_img = Image.open(gt_path).convert('RGB')
        pred_img =Image.open(pred_path).convert('RGB')
        gt_img = gt_img.resize(pred_img.size, Image.BILINEAR)

        gt_tensor = to_tensor(gt_img).unsqueeze(0)
        pred_tensor = to_tensor(pred_img).unsqueeze(0)

        total+=SSIM(gt_tensor, pred_tensor)


    return total / sample_size