import os
import torch
import torch.utils.data as data
import numpy as np
import random
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])

def train_val_list(enhan_images_path, ori_images_path):
    image_list_index = os.listdir(ori_images_path) 
    all_length = len(image_list_index)
    image_list_index = random.sample(image_list_index, all_length)

    image_dataset = []
    for i in image_list_index:  # Add paths and combine them
        image_dataset.append((enhan_images_path + i, ori_images_path + i))

    train_list = image_dataset[:int(all_length*0.9)]
    val_list = image_dataset[int(all_length*0.9):]

    return train_list, val_list

def CLAHE(input_img):
    # Convert the BGR image to YCrCb color space
    ycc_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)
    # Split the channels
    y, cr, cb = cv2.split(ycc_img)
    # Create a CLAHE object (Contrast Limited AHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the luminance channel
    y_clahe = clahe.apply(y)
    # Merge the enhanced Y channel back with Cr and Cb
    ycc_clahe = cv2.merge((y_clahe, cr, cb))
    # Convert back to BGR color space
    output_img = cv2.cvtColor(ycc_clahe, cv2.COLOR_YCrCb2BGR)
    return output_img

class train_val_loader(data.Dataset):

    def __init__(self, enhan_images_path, ori_images_path, mode='train'):

        self.train_list, self.val_list = train_val_list(enhan_images_path, ori_images_path)
        self.mode = mode
        if self.mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_clean_path, data_ori_path = self.data_list[index]

        data_clean = cv2.imread(data_clean_path)
        data_ori = cv2.imread(data_ori_path)
        data_ahe = CLAHE(data_ori)

        data_clean = np.asarray(data_clean) / 255.0
        data_ori = np.asarray(data_ori) / 255.0
        data_ahe = np.asarray(data_ahe) / 255.0

        data_clean = torch.from_numpy(data_clean).float()
        data_ori = torch.from_numpy(data_ori).float()
        data_ahe = torch.from_numpy(data_ahe).float()

        return data_clean.permute(2, 0, 1), data_ori.permute(2, 0, 1), data_ahe.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


class d_train_val_loader(data.Dataset):

    def __init__(self, enhan_images_path, ori_images_path, mode='train'):

        self.train_list, self.val_list = train_val_list(enhan_images_path, ori_images_path)
        self.mode = mode
        if self.mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_clean_path, data_ori_path = self.data_list[index]

        data_gt = cv2.imread(data_clean_path, cv2.IMREAD_GRAYSCALE)
        data_ori = cv2.imread(data_ori_path)
        data_ahe = CLAHE(data_ori)

        data_gt = np.asarray(data_gt) / 255.0
        data_ori = np.asarray(data_ori) / 255.0
        data_ahe = np.asarray(data_ahe) / 255.0

        data_gt = torch.from_numpy(data_gt).unsqueeze(2).float()
        data_ori = torch.from_numpy(data_ori).float()
        data_ahe = torch.from_numpy(data_ahe).float()

        return data_gt.permute(2, 0, 1), data_ori.permute(2, 0, 1), data_ahe.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


class test_loader(data.Dataset):
    def __init__(self, ori_images_path):
        super(test_loader, self).__init__()

        image_list_index = sorted(os.listdir(ori_images_path))
        self.image_dataset = [os.path.join(ori_images_path, x) for x in image_list_index if is_image_file(x)]
        self.all_length = len(self.image_dataset)

    def __len__(self):
        return self.all_length

    def __getitem__(self, index):

        data_ori_path = self.image_dataset[index]
        filename = data_ori_path.split('/')[-1]
        data_ori = cv2.imread(data_ori_path)
        data_ahe = CLAHE(data_ori)

        data_ori = np.asarray(data_ori) / 255.0
        data_ahe = np.asarray(data_ahe) / 255.0

        data_ori = torch.from_numpy(data_ori).float()
        data_ahe = torch.from_numpy(data_ahe).float()

        return data_ori.permute(2, 0, 1), data_ahe.permute(2, 0, 1), filename


def train_val_list_withdepth(enhan_images_path, ori_images_path, depth_image_path):
    image_list_index = os.listdir(ori_images_path)
    all_length = len(image_list_index)
    image_list_index = random.sample(image_list_index, all_length)

    image_dataset = []
    for i in image_list_index:  # Add paths and combine them
        image_dataset.append((enhan_images_path + i, ori_images_path + i, depth_image_path + i))

    train_list = image_dataset[:int(all_length*0.9)]
    val_list = image_dataset[int(all_length*0.9):]

    return train_list, val_list

class train_val_loader_withdepth(data.Dataset):

    def __init__(self, enhan_images_path, ori_images_path, depth_image_path, mode='train'):

        self.train_list, self.val_list = train_val_list_withdepth(enhan_images_path, ori_images_path, depth_image_path)
        self.mode = mode
        if self.mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_clean_path, data_ori_path, data_depth_path = self.data_list[index]

        data_clean = cv2.imread(data_clean_path)
        data_ori = cv2.imread(data_ori_path)
        data_depth = cv2.imread(data_depth_path, cv2.IMREAD_GRAYSCALE)
        data_ahe = CLAHE(data_ori)

        data_clean = np.asarray(data_clean) / 255.0
        data_ori = np.asarray(data_ori) / 255.0
        data_depth = np.asarray(data_depth) / 255.0
        data_ahe = np.asarray(data_ahe) / 255.0

        data_clean = torch.from_numpy(data_clean).float()
        data_ori = torch.from_numpy(data_ori).float()
        data_depth = torch.from_numpy(data_depth).unsqueeze(2).float()
        data_ahe = torch.from_numpy(data_ahe).float()

        return data_clean.permute(2, 0, 1), data_ori.permute(2, 0, 1), data_depth.permute(2, 0, 1), data_ahe.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

class test_loader_withdepth(data.Dataset):
    def __init__(self, ori_images_path, depth_images_path):
        super(test_loader_withdepth, self).__init__()

        image_list_index = sorted(os.listdir(ori_images_path))
        self.image_dataset = [os.path.join(ori_images_path, x) for x in image_list_index if is_image_file(x)]
        self.depth_dataset = [os.path.join(depth_images_path, x) for x in image_list_index if is_image_file(x)]
        self.all_length = len(self.image_dataset)

    def __len__(self):
        return self.all_length

    def __getitem__(self, index):

        data_ori_path = self.image_dataset[index]
        data_depth_path = self.depth_dataset[index]
        filename = data_ori_path.split('/')[-1]

        data_ori = cv2.imread(data_ori_path)
        data_depth = cv2.imread(data_depth_path, cv2.IMREAD_GRAYSCALE)
        data_ahe = CLAHE(data_ori)

        data_ori = np.asarray(data_ori) / 255.0
        data_depth = np.asarray(data_depth) / 255.0
        data_ahe = np.asarray(data_ahe) / 255.0

        data_ori = torch.from_numpy(data_ori).float()
        data_depth = torch.from_numpy(data_depth).unsqueeze(2).float()
        data_ahe = torch.from_numpy(data_ahe).float()

        return data_ori.permute(2, 0, 1), data_depth.permute(2, 0, 1), data_ahe.permute(2, 0, 1), filename