import torch
import random
import os
import math
import torch.nn.functional as f
import lpips
from PIL import Image
from torchvision import transforms

def PSNR(truth, pred):
    mse = f.mse_loss(pred,truth,reduction='mean').item()
    if mse == 0: # same image
        return 100
    return 20*math.log10(1.0/math.sqrt(mse))

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

def evaluate():
    avP = 0.0
    avS = 0.0
    avL = 0.0
    avI = 0.0
    batches = 32
    sample_size = 128
    to_tensor = transforms.ToTensor()
    device= "cuda" if torch.cuda.is_available() else "cpu"

    loss_LPIPS = lpips.LPIPS(net='vgg').to(device)
    loss_LPIPS.eval()

    for x in range(batches):
        p_total = 0.0
        s_total = 0.0
        l_total = 0.0
        i_total = 0.0
        gt_files = sorted(os.listdir("./dataset/test/test_C"))
        gt_files = random.sample(gt_files, sample_size)
        for file in gt_files:
            gt_img_path = os.path.join("./dataset/test/test_C", file)
            pred_img_path = os.path.join("./test_result/shadow_removal_image", file)
            gt_mask_path = os.path.join("./dataset/test/test_B", file)
            pred_mask_path = os.path.join("./test_result/detected_shadow", file)

            gt_img = Image.open(gt_img_path).convert('RGB')
            pred_img = Image.open(pred_img_path).convert('RGB')
            gt_mask = Image.open(gt_mask_path).convert('L')
            pred_mask = Image.open(pred_mask_path).convert('L')


            gt_img = gt_img.resize(pred_img.size, Image.BILINEAR)
            gt_mask = gt_img.resize(pred_mask.size, Image.BILINEAR)

            gt_img_tensor = to_tensor(gt_img).unsqueeze(0).to(device)
            pred_img_tensor = to_tensor(pred_img).unsqueeze(0).to(device)
            gt_mask_tensor = to_tensor(gt_mask).to(device)
            pred_mask_tensor = to_tensor(pred_mask).to(device)

            #PSNR
            p_total += PSNR(gt_img_tensor, pred_img_tensor)

            #SSIM
            s_total += SSIM(gt_img_tensor, pred_img_tensor)
            
            #LPIPS
            with torch.no_grad():
                LPIPS_val = loss_LPIPS(gt_img_tensor, pred_img_tensor)
                l_total += LPIPS_val.item()

            #IoU
            gt_thresh = (gt_mask_tensor > 0.1).float()
            pred_thresh = (pred_mask_tensor > 0.1).float()

            intersection = (gt_thresh * pred_thresh).sum()
            union = ((gt_thresh + pred_thresh) > 0).float().sum()
            iou = (intersection / union).item()
            i_total += iou

            #add to averages
        print("Batch ", x)
        print("PSNR: ", p_total/sample_size)
        print("SSIM: ", s_total/sample_size)
        print("LPIPS: ", l_total/sample_size)
        print("IoU: ", i_total/sample_size)
        avP += p_total/sample_size
        avS += s_total/sample_size
        avL += l_total/sample_size
        avI += i_total/sample_size

    
    print("PSNR: ", avP/batches)
    print("SSIM: ", avS/batches)
    print("LPIPS", avL/batches)
    print("IoU: ", avI/batches)


if __name__ == "__main__":
    evaluate()