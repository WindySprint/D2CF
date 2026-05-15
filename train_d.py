import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from models import d_network
from utils import dataloader, losses

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
# Input Parameters
parser.add_argument('--net_name', type=str, default="d_net")
parser.add_argument('--depth_images_path', type=str, default="../../../share_ssd/UIE/datasets/12/train/USOD10K/gt/")
parser.add_argument('--ori_images_path', type=str, default="../../../share_ssd/UIE/datasets/12/train/USOD10K/input/")
parser.add_argument('--n_feats', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--val_batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--checkpoint_path', type=str, default="checkpoints/")
parser.add_argument('--cudaid', type=str, default="0",help="choose cuda device id).")
config = parser.parse_args()

def train(config):
    print("gpu_id:", config.cudaid)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid

    d_net = d_network.RCNet(config.n_feats).cuda()

    if not os.path.exists(os.path.join(config.checkpoint_path, config.net_name)):
        os.mkdir(os.path.join(config.checkpoint_path, config.net_name))

    if len(config.cudaid) > 1:
        cudaid_list = config.cudaid.split(",")
        cudaid_list = [int(x) for x in cudaid_list]
        device_ids = [i for i in cudaid_list]
        d_net = nn.DataParallel(d_net, device_ids=device_ids)

    criterion_mse = losses.MSE_Loss()

    train_dataset = dataloader.d_train_val_loader(config.depth_images_path,config.ori_images_path)
    val_dataset = dataloader.d_train_val_loader(config.depth_images_path,config.ori_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)

    ######### Adam optimizer ###########
    optimizer = optim.Adam(d_net.parameters(), lr=config.lr)
    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs - warmup_epochs,
                                                            eta_min=config.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    d_net.train()

    # Record best index and corresponding epoch
    best_psnr = 0
    best_epoch = 0

    for epoch in range(1, config.num_epochs+1):
        epoch_start_time = time.time()
        # Record train loss and validation index
        train_loss = []
        val_psnr = []
        print("*" * 30 + "The %i epoch" % epoch + "*" * 30+'\n')
        
        for _, (img_gt, img_ori, img_ahe) in enumerate(tqdm(train_loader)):
            img_gt = img_gt.cuda()
            img_ori = img_ori.cuda()
            img_ahe = img_ahe.cuda()

            try:
                img_d = d_net(img_ori, img_ahe)
                # l1_loss = criterion_l1(img_gt, img_d)
                mse_loss = criterion_mse(img_gt, img_d)
                # sum_loss = l1_loss + 3*mse_loss
                sum_loss = mse_loss

                train_loss.append(sum_loss.item())
                optimizer.zero_grad()
                sum_loss.backward()
                torch.nn.utils.clip_grad_norm_(d_net.parameters(), config.grad_clip_norm)
                optimizer.step()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

        with open(os.path.join(config.checkpoint_path, config.net_name, "loss.log"), "a+", encoding="utf-8") as f:
            s = "The %i Epoch mean_loss is :%f" % (epoch, np.mean(train_loss)) + "\n"
            f.write(s)

        # Validation Stage
        with torch.no_grad():
            for _, (img_gt, img_ori, img_ahe) in enumerate(val_loader):
                img_gt = img_gt.cuda()
                img_ori = img_ori.cuda()
                img_ahe = img_ahe.cuda()
                img_d = d_net(img_ori, img_ahe)

                psnr = losses.torchPSNR(img_gt, img_d)
                val_psnr.append(psnr.item())

        val_psnr = np.mean(np.array(val_psnr))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            torch.save({'state_dict': d_net.state_dict()}, os.path.join(config.checkpoint_path, config.net_name, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
              (epoch, val_psnr, best_epoch, best_psnr))

        with open(os.path.join(config.checkpoint_path, config.net_name, "val_PSNR.log"), "a+", encoding="utf-8") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                    (epoch, val_psnr, best_epoch, best_psnr) + "\n")

        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format
              (epoch, time.time() - epoch_start_time, np.mean(train_loss), scheduler.get_lr()[0]))

        scheduler.step()
        torch.save({'state_dict': d_net.state_dict()}, os.path.join(config.checkpoint_path, config.net_name, "model_latest.pth"))

if __name__ == "__main__":
    start_time = time.time()
    train(config)
    e = time.time()
    print("train_time:"+str(time.time()-start_time))
