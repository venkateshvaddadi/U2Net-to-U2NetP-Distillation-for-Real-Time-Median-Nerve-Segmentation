#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:31:54 2026

@author: venkatesh
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.optim as optim
import time
import tqdm

#%%
from CTS_dataset import mydataloader
from loss.diceloss import *
from models.u2_net_model.u2net import U2NET,U2NETP

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
import torch
import torch.nn as nn
import pytorch_ssim







# -----------------------------
# Stable Soft IoU Loss
# -----------------------------
class SoftIoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        loss = 1 - iou

        return loss.mean()


# -----------------------------
# Base Losses
# -----------------------------
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = SoftIoULoss()

mse_loss = nn.MSELoss()

# -----------------------------
# GT Supervision Loss
# -----------------------------
def bce_ssim_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + ssim_out + iou_out
    return loss


def multi_scale_gt_loss(d0, d1, d2, d3, d4, d5, d6, target):
    loss0 = bce_ssim_iou_loss(d0, target)
    loss1 = bce_ssim_iou_loss(d1, target)
    loss2 = bce_ssim_iou_loss(d2, target)
    loss3 = bce_ssim_iou_loss(d3, target)
    loss4 = bce_ssim_iou_loss(d4, target)
    loss5 = bce_ssim_iou_loss(d5, target)
    loss6 = bce_ssim_iou_loss(d6, target)

    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return total_loss


# -----------------------------
# KD Loss (Student vs Teacher)
# -----------------------------
def kd_loss_per_output(s, t, lambda_mse=0.7, lambda_iou=0.3):
    """
    s: student prediction (probability map)
    t: teacher prediction (probability map)
    """

    # Pixel-wise alignment
    loss_mse = mse_loss(s, t)

    # Structure alignment (soft IoU)
    loss_iou = iou_loss(s, t)

    return lambda_mse * loss_mse + lambda_iou * loss_iou


# -----------------------------
# Final Distillation Loss
# -----------------------------
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, lambda_mse=0.7, lambda_iou=0.3):
        """
        alpha: weight for GT vs KD
        lambda_mse: weight for MSE in KD
        lambda_iou: weight for IoU in KD
        """
        super().__init__()

        self.alpha = alpha
        self.lambda_mse = lambda_mse
        self.lambda_iou = lambda_iou

    def forward(self, s_outputs, t_outputs, target):

        # -------------------------
        # 1. Ground Truth Loss
        # -------------------------
        loss_gt = multi_scale_gt_loss(*s_outputs, target)

        # -------------------------
        # 2. Knowledge Distillation Loss
        # -------------------------
        loss_kd = 0

        for s, t in zip(s_outputs, t_outputs):
            loss_kd += kd_loss_per_output(
                s, t,
                self.lambda_mse,
                self.lambda_iou
            )

        # -------------------------
        # 3. Final Loss
        # -------------------------
        total_loss = self.alpha * loss_gt + (1 - self.alpha) * loss_kd

        return total_loss


#%%


    
batch_size=16
learning_rate=1e-4
no_patients=100
gpu_id=0
restore=False
alpha_for_GT_vs_distilling=0.7
#%%
from datetime import datetime
import time
import os

print('*******************************************************')
start_time = time.time()

# -------------------------------
# EXPERIMENT CONFIG
# -------------------------------
teacher_name = "U2NET"
student_name = "U2NETP"
experiment_type = "distillation"
data_setting = "100_percent_data"   # change: 25/50/75/full
loss_type = "bce_ssim_iou_multiscale"

# -------------------------------
# BASE FOLDER
# -------------------------------
experiments_folder = "savedModels/distillation_experiments"

# -------------------------------
# EXPERIMENT NAME
# -------------------------------
timestamp = datetime.now().strftime("%b_%d_%I_%M_%p")

experiment_name = (
    f"{timestamp}_"
    f"{teacher_name}_to_{student_name}_"
    f"{experiment_type}_"
    f"{data_setting}_"
    f"{loss_type}"
)

# -------------------------------
# FINAL DIRECTORY
# -------------------------------
directory = os.path.join(experiments_folder, experiment_name)

print("Experiment Directory:", directory)
print("Model will be saved to:", directory)

# -------------------------------
# CREATE DIRECTORY
# -------------------------------
os.makedirs(directory, exist_ok=True)
#%%
#%%
import torch

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD TEACHER MODEL
# -----------------------------
teacher_model_path = "../savedModels/full_data_experiemnts/muti_bce_loss_fusion//Jul_29_06_42_pm_data_representation_model_u2net/CTS_U_net_model_5_.pth"

teacher = U2NET(3, 1)

state_dict = torch.load(teacher_model_path, map_location="cpu")

# -----------------------------
# HANDLE DATAPARALLEL
# -----------------------------
if list(state_dict.keys())[0].startswith("module."):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    state_dict = new_state_dict

teacher.load_state_dict(state_dict)

# -----------------------------
# SET MODE
# -----------------------------
teacher.to(device)
teacher.eval()

# -----------------------------
# FREEZE TEACHER
# -----------------------------
for param in teacher.parameters():
    param.requires_grad = False

print("✅ Teacher model loaded successfully")
#%%
student = U2NETP(3, 1)
student.train()
#%%

teacher.to(device)
student.to(device)

#%%

data_path='../../data_making/aster_updated_data_nov_09_2022_with_flip/'
patients_path_csv=data_path+"/csv_files/patients_list_100.csv"

tloader = mydataloader(data_path, '../..//data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/patients_list_1_99.csv', 
                       '../..//data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/data_with_v_h_1_79_train.csv')
train_loader = DataLoader(tloader, batch_size = batch_size, shuffle=True, num_workers=1)








vloader = mydataloader(data_path, '../../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/patients_list_1_99.csv', 
                       '../../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/data_with_v_h_80_89_validation.csv')
val_loader = DataLoader(vloader, batch_size = batch_size, shuffle=True, num_workers=1)




import logging

no_train_batches=len(train_loader)
no_val_batches=len(val_loader)
print(no_train_batches,no_val_batches)

logging.warning('No training samples:'+str(batch_size*len(train_loader)))
logging.warning('No validation samples:'+str(batch_size*len(val_loader)))

logging.warning('no traiiiining batches:'+str(len(train_loader)))
logging.warning('no validation batches:'+str(len(val_loader)))
#%%
criterion = DistillationLoss(alpha=alpha_for_GT_vs_distilling)
optimizer = optim.Adam(student.parameters(), lr=1e-4)



#%%
def compute_metrics(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()

    # Ensure same shape
    assert pred.shape == target.shape, "Shape mismatch!"

    # Get dimensions dynamically (ignore batch dim)
    dims = tuple(range(1, pred.dim()))
    
    
    TP = (pred * target).sum(dim=dims)
    FP = (pred * (1 - target)).sum(dim=dims)
    FN = ((1 - pred) * target).sum(dim=dims)

    precision = (TP + smooth) / (TP + FP + smooth)
    recall    = (TP + smooth) / (TP + FN + smooth)
    dice      = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou       = (TP + smooth) / (TP + FP + FN + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),   # ✅ NEW
        "dice": dice.mean().item(),
        "iou": iou.mean().item()
    }
#%%
import pandas as pd

# store logs
training_log = []
csv_path = os.path.join(directory, "training_metrics.csv")

#%%
num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0

    for i, data in tqdm.tqdm(enumerate(train_loader)):
        
        
        image, mask, _, _ = data

        image = image.float() / 255.0
        mask = mask.float() / 255.0

        image = image.to(device)
        mask = mask.to(device)

        # -------------------------
        # Teacher Forward (NO GRAD)
        # -------------------------
        # Teacher forward
        with torch.no_grad():
            t_outputs = teacher(image)   # (d0...d6)
        
        # Student forward
        s_outputs = student(image)       # (d0...d6)
        
        # Loss
        loss = criterion(s_outputs, t_outputs, mask)
        # -------------------------
        # Backprop
        # -------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss / len(train_loader)}")

    
    student.eval()

    val_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,   # ✅ NEW
        "dice": 0,
        "iou": 0
    }
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            image, mask, _, _ = data
    
            image = image.float() / 255.0
            mask = mask.float() / 255.0
    
            image = image.to(device)
            mask = mask.to(device)
    
            outputs = student(image)
            pred = outputs[0]   # d0
    
            metrics = compute_metrics(pred, mask)
    
            for k in val_metrics:
                val_metrics[k] += metrics[k]
    
    # Average
    for k in val_metrics:
        val_metrics[k] /= len(val_loader)
    
    print(f"Epoch {epoch} | Dice: {val_metrics['dice']:.4f} | "
          f"F1: {val_metrics['f1']:.4f} | "
          f"Precision: {val_metrics['precision']:.4f} | "
          f"Recall: {val_metrics['recall']:.4f} | "
          f"IoU: {val_metrics['iou']:.4f}")

    # Save model
    save_path = os.path.join(directory, f"u2netp_distilled_epoch_{epoch}.pth")
    torch.save(student.state_dict(), save_path)    
    
    
    avg_train_loss = total_loss / len(train_loader)
    
    # -------------------------
    # STORE METRICS
    # -------------------------
    epoch_log = {
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_dice": val_metrics["dice"],
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_iou": val_metrics["iou"]
    }
    
    training_log.append(epoch_log)
    
    # -------------------------
    # SAVE TO CSV (overwrite each epoch)
    # -------------------------
    df = pd.DataFrame(training_log)
    df.to_csv(csv_path, index=False)
    
    print(f"📊 Metrics saved to {csv_path}")    
    
    
    
    
    
    
    
    
    
    
    