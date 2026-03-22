#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QAT + Knowledge Distillation Training
U2Net (Teacher) → U2NetP (Student)
"""

# -----------------------------
# IMPORTS
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.quantization as tq
import tqdm
import os
import pandas as pd
from datetime import datetime
import logging

# -----------------------------
# CUSTOM IMPORTS
# -----------------------------
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

# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    dims = tuple(range(1, pred.dim()))

    TP = (pred * target).sum(dim=dims)
    FP = (pred * (1 - target)).sum(dim=dims)
    FN = ((1 - pred) * target).sum(dim=dims)

    precision = (TP + smooth) / (TP + FP + smooth)
    recall    = (TP + smooth) / (TP + FN + smooth)
    dice      = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou       = (TP + smooth) / (TP + FP + FN + smooth)
    f1        = 2 * (precision * recall) / (precision + recall + smooth)

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1,
        "dice": dice.mean().item(),
        "iou": iou.mean().item()
    }

# -----------------------------
# LOAD TEACHER
# -----------------------------
teacher_model_path = "../savedModels/full_data_experiemnts/muti_bce_loss_fusion//Jul_29_06_42_pm_data_representation_model_u2net/CTS_U_net_model_5_.pth"

teacher = U2NET(3, 1)
state_dict = torch.load(teacher_model_path, map_location="cpu")

# Handle DataParallel
if list(state_dict.keys())[0].startswith("module."):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    state_dict = new_state_dict

teacher.load_state_dict(state_dict)
teacher.to(device)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

print("✅ Teacher loaded")

# -----------------------------
# STUDENT (QAT)
# -----------------------------
student = U2NETP(3, 1)

# Skip final layer quantization
try:
    student.outconv.qconfig = None
except:
    pass

# QAT config
student.qconfig = tq.get_default_qat_qconfig('fbgemm')
tq.prepare_qat(student, inplace=True)

student.to(device)
student.train()

print("✅ QAT enabled")

# -----------------------------
# DATA
# -----------------------------
data_path = "../../data_making/aster_updated_data_nov_09_2022_with_flip/"

train_loader = DataLoader(
    mydataloader(
        data_path,
        data_path + "csv_files/full_data_csv/patients_list_1_99.csv",
        data_path + "csv_files/full_data_csv/data_with_v_h_1_79_train.csv"
    ),
    batch_size=8, shuffle=True, num_workers=2
)

val_loader = DataLoader(
    mydataloader(
        data_path,
        data_path + "csv_files/full_data_csv/patients_list_1_99.csv",
        data_path + "csv_files/full_data_csv/data_with_v_h_80_89_validation.csv"
    ),
    batch_size=8, shuffle=False, num_workers=2
)

# -----------------------------
# OPTIMIZER
# -----------------------------
criterion = DistillationLoss(alpha=0.7)
optimizer = optim.Adam(student.parameters(), lr=1e-4)

# -----------------------------
# LOGGING
# -----------------------------
timestamp = datetime.now().strftime("%b_%d_%H_%M")
save_dir = f"savedModels/QAT_KD/{timestamp}"
os.makedirs(save_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "metrics.csv")
training_log = []

# -----------------------------
# TRAINING
# -----------------------------
num_epochs = 50

for epoch in range(num_epochs):

    student.train()
    total_loss = 0

    # Freeze observers after 70%
    if epoch == int(num_epochs * 0.7):
        print("🔒 Freezing observers...")
        student.apply(torch.quantization.disable_observer)
        student.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    for image, mask, _, _ in tqdm.tqdm(train_loader):

        image = image.float().to(device) / 255.0
        mask  = mask.float().to(device) / 255.0

        with torch.no_grad():
            t_outputs = teacher(image)

        s_outputs = student(image)

        loss = criterion(s_outputs, t_outputs, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # -----------------------------
    # VALIDATION
    # -----------------------------
    student.eval()
    val_metrics = {"precision":0,"recall":0,"f1":0,"dice":0,"iou":0}

    with torch.no_grad():
        for image, mask, _, _ in val_loader:

            image = image.float().to(device) / 255.0
            mask  = mask.float().to(device) / 255.0

            outputs = student(image)
            pred = outputs[0]

            metrics = compute_metrics(pred, mask)

            for k in val_metrics:
                val_metrics[k] += metrics[k]

    for k in val_metrics:
        val_metrics[k] /= len(val_loader)

    print(f"Epoch {epoch} | Loss {avg_loss:.4f} | Dice {val_metrics['dice']:.4f}")

    # Save FP32 model
    torch.save(student.state_dict(),
               os.path.join(save_dir, f"model_epoch_{epoch}.pth"))

    # CSV logging
    log = {"epoch": epoch, "loss": avg_loss, **val_metrics}
    training_log.append(log)
    pd.DataFrame(training_log).to_csv(csv_path, index=False)

# -----------------------------
# CONVERT TO INT8
# -----------------------------
print("⚡ Converting to INT8...")

student.eval()
student.cpu()

quantized_model = tq.convert(student, inplace=False)

torch.save(quantized_model.state_dict(),
           os.path.join(save_dir, "model_int8.pth"))

print("✅ INT8 model saved")

# -----------------------------
# TORCHSCRIPT
# -----------------------------
scripted = torch.jit.script(quantized_model)
scripted.save(os.path.join(save_dir, "model_mobile.pt"))

print("📱 Mobile model ready!")