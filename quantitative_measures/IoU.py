import os
from PIL import Image
from torchvision import transforms
import random
def compute_IoU(truth_folder, pred_folder, sample_size):
    to_tensor = transforms.ToTensor()
    total = 0.0
    gt_files = sorted(os.listdir(truth_folder))
    gt_files = random.sample(gt_files, sample_size)

    for fname in gt_files:
        gt_path = os.path.join(truth_folder, fname)
        pred_path = os.path.join(pred_folder, fname)

        gt_img = Image.open(gt_path).convert('L')
        pred_img = Image.open(pred_path).convert('L')


        gt_img = gt_img.resize(pred_img.size, Image.BILINEAR)

        gt_tensor = to_tensor(gt_img)
        pred_tensor = to_tensor(pred_img)

        gt_thresh = ((gt_tensor > 0.1).float())
        pred_thresh = (pred_tensor > 0.1).float()

        intersection = (gt_thresh * pred_thresh).sum()
        union = ((gt_thresh + pred_thresh) > 0).float().sum()
        iou = (intersection / union).item()
        total += iou

    return total / sample_size