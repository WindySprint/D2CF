import os
import time
import argparse
import torch
import torch.nn as nn

import cv2
import numpy as np
import torch.optim
from tqdm import tqdm
from collections import OrderedDict

from models import network, d_network
from utils import dataloader

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--d_net_name', type=str, default="d_net")
parser.add_argument('--net_name', type=str, default="net")
parser.add_argument('--dataset_name', type=str, default="UVTD")
parser.add_argument('--ori_images_path', type=str, default="../../../share_ssd/UIE/datasets/test")
parser.add_argument('--n_feats', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
parser.add_argument('--result_path', type=str, default="./results/")
parser.add_argument('--cudaid', type=str, default="0", help="choose cuda device id 0-7).")

config = parser.parse_args()
print("gpu_id:", config.cudaid)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def test(config):
    d_net = d_network.RCNet(config.n_feats).cuda()
    load_checkpoint(d_net, os.path.join(config.checkpoint_path, config.d_net_name, 'model_best.pth'))
    enhan_net = network.CPGNet(config.n_feats).cuda()
    load_checkpoint(enhan_net, os.path.join(config.checkpoint_path, config.net_name, 'model_best.pth'))

    if len(config.cudaid) > 1:
        cudaid_list = config.cudaid.split(",")
        cudaid_list = [int(x) for x in cudaid_list]
        device_ids = [i for i in cudaid_list]
        d_net = nn.DataParallel(d_net, device_ids=device_ids)
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)
        
    print(os.path.join(config.ori_images_path, config.dataset_name))
    test_dataset = dataloader.test_loader(os.path.join(config.ori_images_path, config.dataset_name))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                               num_workers=config.num_workers, drop_last=False, pin_memory=True)
    # create results folds
    if not os.path.exists(os.path.join(config.result_path, config.net_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name))

    if not os.path.exists(os.path.join(config.result_path, config.net_name, config.dataset_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name, config.dataset_name))

    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)

    d_net.eval()
    enhan_net.eval()

    with torch.no_grad():
        for _, (img_ori, img_ahe, filenames) in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_ori = img_ori.cuda()
            img_ahe = img_ahe.cuda()
            img_d = d_net(img_ori, img_ahe)
            img_en = enhan_net(img_ori, img_ahe, img_d)
            for i in range(len(img_en)):
                en_img = img_en[i, :, :, :].cpu().detach().numpy()
                en_img = np.transpose(en_img, (1, 2, 0))
                cv2.imwrite(os.path.join(result_dir, filenames[i]), en_img* 255.0)

    # from thop import profile
    # flops1, params1 = profile(d_net, inputs=(img_ori, img_ahe))
    # print('flops1: %.4f G, params1: %.4f M' % (flops1, params1))
    # flops2, params2 = profile(enhan_net, inputs=(img_ori, img_ahe, img_d))
    # print('flops2: %.4f G, params2: %.4f M' % (flops2, params2))
    # print('all_flops: %.4f G, all_params: %.4f M' % ((flops1+flops2)/(10e+8), (params1+params2)/(10e+5)))

if __name__ == '__main__':
    start_time = time.time()
    test(config)
    print("test_time:"+str((time.time()-start_time)))
