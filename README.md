# ST-CGAN: Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal with PyTorch

This repository is an alteration to the code [ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks](https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks), which itself is an unofficial implementation of  [Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal](https://arxiv.org/abs/1712.02478) [Wang+, **CVPR** 2018] with PyTorch.

## Requirements
* Python3.x
* PyTorch 1.5.0
* pillow
* matplotlib

## Usage
Datasets have been altered to include both the ISTD and SRD. Download the dataset from this link: https://drive.google.com/file/d/1Go2zQEFlF8htdkD3WYd3SjFxRH6aWIR_/view?usp=drive_link

Once the data has been downloaded, unzip into your main project folder. Directories should be:
* ./dataset/train/...
* ./dataset/test/... 

Warning: the dataset is about 14.1GB in full. 

Shadow masks for the SRD can be regenerated under different hyperparameters by adjusting and running create_shadow_map.py


### Training
```
python3 train.py
```
### Testing
When Testing images from ISTD dataset.
```
python3 test.py -l <checkpoint number>
```
When you would like to test your own image.
```
python3 test.py -l <checkpoint number> -i <image_path> -o <out_path>
```

### Evaluation of generated images
Quantitative analysis of images can be done by running eval.py. This will only work after generating images with test.py

## Results
Here is a result from test sets.
![](https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks/blob/master/result/91-3.png)
(Left to right: input, ground truth, shadow removal, ground truth shadow, shadow detection)

### Shadow Detection
Here are some results from validation set.
![](https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks/blob/master/result/detected_shadow.jpg)
(Top to bottom: ground truth, shadow detection)

### Shadow Removal
Here are some results from validation set.
![](https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks/blob/master/result/shadow_removal.jpg)
(Top to bottom: input, ground truth, shadow removal)

## Trained model
You can download from [here](https://drive.google.com/drive/folders/1J1l21k5AoUXHxic-Bj3eXBFP--YzjFXO?usp=sharing).

## References
* Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal, Jifeng Wang<sup>∗</sup>, Xiang Li<sup>∗</sup>, Le Hui, Jian Yang, **Nanjing University of Science and Technology**, [[arXiv]](https://arxiv.org/abs/1712.02478)
* https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks
