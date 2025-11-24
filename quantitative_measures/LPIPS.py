import os
from PIL import Image
from torchvision import transforms
import torch
import lpips
import random

def compute_LPIPS(truth_folder,pred_folder, device, sample_size):
    loss_LPIPS = lpips.LPIPS(net='vgg').to(device)
    loss_LPIPS.eval()
    total_LPIPS = 0.0
    
    to_tensor = transforms.ToTensor()

    gt_files = sorted(os.listdir(truth_folder))
    
    gt_files = random.sample(gt_files, sample_size)
    for fname in gt_files:
        gt_path = os.path.join(truth_folder, fname)
        pred_path = os.path.join(pred_folder, fname)

        gt_img = Image.open(gt_path).convert('RGB')
        pred_img = Image.open(pred_path).convert('RGB')
        gt_img = gt_img.resize(pred_img.size, Image.BILINEAR)

        gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device)
        pred_tensor = to_tensor(pred_img).unsqueeze(0).to(device)

        with torch.no_grad():
            LPIPS_val = loss_LPIPS(gt_tensor, pred_tensor)
            total_LPIPS += LPIPS_val.item()


    return total_LPIPS/sample_size

