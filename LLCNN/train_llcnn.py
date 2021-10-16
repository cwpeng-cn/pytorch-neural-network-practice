#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/11 11:33
# @Author  : CongXiaofeng
# @File    : train.py
# @Software: PyCharm

import save
import torch
import os
from utils import make_project_dir
from loss_utils import LossWriter
from loss_utils import SSIMLoss
from llcnn_net import LLCNN
import torch.optim as optim
import torch.nn as nn
from datasets import EnahanceDatasets

# 设置LLCNN训练超参数

BETA1 = 0.9
BETA2 = 0.999
DATA_ROOT = "gamma_dataset"
INPUT_DIR_NAME = "dark"
LABEL_DIR_NAME = "clear"
LR = 0.001
BATCH_SIZE = 8
W_L1 = 1
W_L2 = 1
W_SMO_L1 = 1
W_SSIM = 1
H_FLIP = True
V_FLIP = True
RESULTS_DIR = "results"
EPOCHS = 50
IMAGE_SIZE = 256
IMG_SAVE_FREQ = 100
PTH_SAVE_FREQ = 2

VAL_BATCH_SIZE = 1
VAL_FREQ = 1


device = torch.device("cuda")

# 定义训练集的dataloader和测试集的dataloader
train_dataset = EnahanceDatasets(IMAGE_SIZE, DATA_ROOT, INPUT_DIR_NAME, LABEL_DIR_NAME, H_FLIP, V_FLIP, train=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
val_dataset = EnahanceDatasets(IMAGE_SIZE, DATA_ROOT, INPUT_DIR_NAME, LABEL_DIR_NAME, H_FLIP, V_FLIP, train=False)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=VAL_BATCH_SIZE,
                                             shuffle=True)


# 定义训练LLCNN可以使用的四种损失函数
L1_func = nn.L1Loss()
L2_func = nn.MSELoss()
SMO_L1_func = nn.SmoothL1Loss()
SSIM_func = SSIMLoss()

# 定义LLCNN网络
llcnn = LLCNN().to(device)
# 定义优化器，将LLCNN网络参数绑定到优化器
optimizer = optim.Adam(params=llcnn.parameters(),
                       lr=LR,
                       betas=(BETA1, BETA2))

make_project_dir(RESULTS_DIR, RESULTS_DIR)
# 记录训练过程的损失函数值
loss_writer = LossWriter(os.path.join(RESULTS_DIR, "loss"))


def train():
    iteration = 0
    for epo in range(1, EPOCHS):
        for data in train_loader:
            dark = data["dark"].to(device)
            clear = data["clear"].to(device)
            
            # 前向计算，获得预测的清晰图像
            predict_clear = llcnn(dark)

            # 根据设定的权重，计算四种损失
            l1_loss = W_L1 * L1_func(predict_clear, clear)
            l2_loss = W_L2 * L2_func(predict_clear, clear)
            smo_l1_loss = W_SMO_L1 * SMO_L1_func(predict_clear, clear)
            ssim_loss = W_SSIM * SSIM_func(predict_clear, clear)

            # 计算LLCNN前向计算过程的总损失函数
            loss = l1_loss + l2_loss + smo_l1_loss + ssim_loss

            # 更新网络参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 将损失值写入文件
            loss_writer.add("loss", loss.item(), iteration)
            loss_writer.add("l1 loss", l1_loss.item(), iteration)
            loss_writer.add("l2 loss", l2_loss.item(), iteration)
            loss_writer.add("smooth l1 loss", smo_l1_loss.item(), iteration)
            loss_writer.add("ssim loss", ssim_loss.item(), iteration)

            iteration += 1

            # 在控制台输出每次迭代的损失值
            print("Iter: {}, Total Loss: {:.4f}, L1 Loss: {:.4f}, "
                  "L2 Loss: {:.4f}, Smooth L1 Loss: {:.4f},"
                  "SSIM Loss: {:.4f}".format(iteration,loss.item(),
                                         l1_loss.item(),
                                         l2_loss.item(),
                                         smo_l1_loss.item(),
                                         ssim_loss.item()))

            # 每隔一定的迭代，保存训练过程的暗光增强效果
            # 便于定性分析训练效果
            if iteration % IMG_SAVE_FREQ == 0:
                train_patch = torch.cat((dark, predict_clear, clear), dim=3)
                save.save_image(train_patch[0],
                                out_name=os.path.join(RESULTS_DIR, "train_images",
                                                      str(iteration) + ".png"))

        # 保存LLCNN模型文件
        if epo % PTH_SAVE_FREQ == 0:
            torch.save(llcnn.state_dict(), os.path.join(RESULTS_DIR,
                                                        "pth", str(epo) + ".pth"))

        # 将模型切换到eval模式，评估暗光增强模型在验证集上的效果
        if epo % VAL_FREQ == 0:
            llcnn.eval()
            with torch.no_grad():
                for data in val_loader:
                    dark = data["dark"].to(device)
                    clear = data["clear"].to(device)
                    img_name = data["img_name"]
                    predict_clear = llcnn(dark)
                    val_patch = torch.cat((dark, predict_clear, clear), dim=3)
                    save.save_image(val_patch[0],
                                    out_name=os.path.join(RESULTS_DIR,
                                                          "val_images", img_name[0]))
            llcnn.train()


if __name__ == "__main__":
    train()
