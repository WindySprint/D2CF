# CGFNet
This is the project of paper "D2CF: Depth-Guided Degradation-Aware Cross-Fusion Framework for Underwater Image Enhancement".

# 1. Abstract 
Due to light absorption and scattering effects, underwater images often suffer from color distortion, low contrast, and blurred details. These degradations are spatially non-uniform and continuously distributed across the scene, typically exhibiting more severe attenuation in distant regions than in nearby regions. Most existing enhancement methods adopt uniform processing strategies, which overlook such spatially varying degradation and lead to inconsistent visual quality. To address these challenges, we propose D2CF, a novel cross-fusion framework for underwater image enhancement. This framework first generates depth maps by combining enhancement prompts with region-level contrast cues, providing explicit physical guidance associated with the degradation distribution. Subsequently, D2CF employs depth-guided regional contrast fusion learning to capture continuous spatial degradation variations, while mitigating cross-regional interference through local-global collaborative optimization, thereby facilitating precise color correction and detail restoration. Extensive experiments on public underwater datasets demonstrate that D2CF effectively handles various non-uniform degradation scenarios, outperforming existing state-of-the-art methods in both subjective evaluations and objective metrics. The code is available at https://github.com/WindySprint/D2CF.

## Environment
```
1. Python 3.10.13
2. PyTorch 2.1.1
3. Torchvision 0.16.1
4. OpenCV-Python 4.9.0.80
5. NumPy 1.26.3
6. Mamba-ssm 1.2.0.post1
```

## Checkpoints
in the 'checkpoints' foloder.

## Test
if you use pre-trained d_net model to enahnce underwater images
```
1. Clone repo
2. Download 'trained_model' folder and place it in repo
3. Put the images in your folder path A
4. Change 'ori_images_path' to A and 'result_path' in test.py
5. Change 'dataset_name'
6. Run test.py
7. Find results in 'result_path'
```

if you have depth images to enahnce underwater images
```
1. Clone repo
2. Download 'trained_model' folder and place it in repo
3. Put the images in your folder path A
4. Change 'ori_images_path' to A and 'result_path' in test_withdepth.py
5. Change 'dataset_name'
6. Change 'depth_images_path' to A and 'result_path' in test_withdepth.py
7. Run test_withdepth.py
8. Find results in 'result_path'
```

## Train
```
Train d_net
1. Put the orignal images and the depth images in your folder path A and B
2. Change 'ori_images_path' to A and 'depth_images_path' to B in train_d.py
3. Run train_d.py
4. Find trained d_net in 'checkpoint_path'/'d_net_name'
```

if you use pre-trained d_net model to train net
```
Train net with pre-trained model for generate enhanced image
1. Put the orignal images and the GT images in your folder path A and B
2. Change 'ori_images_path' to A and 'enhan_images_path' to B in train.py
3. Ensure 'd_net_name' is consistent with the folder names of d_net
4. Change 'net_name' in train.py
5. Run train.py
6. Find trained net in 'checkpoint_path'/'net_name'
```

if you have depth images to train net
```
Train net with depth image for generate enhanced images
1. Put the orignal images and the GT images in your folder path A and B
2. Change 'ori_images_path' to A and 'enhan_images_path' to B in train_withdepth.py
3. Change 'depth_images_path' to your path in train_withdepth.py
4. Change 'net_name' in train.py
5. Run train_withdepth.py
6. Find trained net in 'checkpoint_path'/'net_name'
```

## Contact
If you have any questions, please contact: Zhixiong Huang: hzxcyanwind@mail.dlut.edu.cn
