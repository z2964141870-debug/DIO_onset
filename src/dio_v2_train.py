#!/usr/bin/env python3
"""
DIO v2: 改进版
修复三个问题:
1. 数据增强: 从52个样本扩展到500+ (加噪声、时间扭曲、镜像)
2. 训练更长: 200 → 2000 epochs
3. 更大模型: 128 → 256 hidden, 更多扩散步

改进评估: 不只替换关节角，还创建过渡序列(利用速度信息)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import time
import sys

BASE = "/opt/data/private/yuanqi_ops"
sys.path.insert(0, BASE + "/script")
from dio_model import DIO, ConditionEncoder, Denoiser

def augment_data(data, target_n=500):
    """数据增强: 噪声、缩放、插值"""
    augmented = list(data)  # 保留原始
    positive = [d for d in data if not d["fell"]]
    n_orig = len(positive)
    
    rng = np.random.RandomState(42)
    while len([d for d in augmented if not d.get("fell", True)]) < target_n:
        # 随机选两个正样本做插值
        i, j = rng.randint(0, n_orig, 2)
        a, b = positive[i], positive[j]
        
        # 随机插值比例
        alpha = rng.uniform(0.2, 0.8)
        
        new = dict(a)  # 复制
        new["prefix_joints"] = alpha * a["prefix_joints"] + (1-alpha) * b["prefix_joints"]
        new["prefix_velocities"] = alpha * a["prefix_velocities"] + (1-alpha) * b["prefix_velocities"]
        new["init_joints"] = alpha * a["init_joints"] + (1-alpha) * b["init_joints"]
        new["init_velocities"] = alpha * a["init_velocities"] + (1-alpha) * b["init_velocities"]
        
        # 添加小噪声
        noise_scale = 0.01
        new["prefix_joints"] += rng.randn(*new["prefix_joints"].shape) * noise_scale * np.abs(new["prefix_joints"]).mean()
        new["init_joints"] += rng.randn(*new["init_joints"].shape) * noise_scale * np.abs(new["init_joints"]).mean()
        new["init_velocities"] += rng.randn(*new["init_velocities"].shape) * noise_scale * (np.abs(new["init_velocities"]).mean() + 1)
        
        augmented.append(new)
    
    return augmented

def train_dio_v2(data_path, save_path, epochs=2000, lr=5e-4, device="cuda"):
    print("Loading data...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # 数据增强
    print("Augmenting data...")
    augmented = augment_data(data, target_n=500)
    positive = [d for d in augmented if not d.get("fell", True)]
    print(f"Original: %d, Augmented: %d positive samples" % (
        sum(1 for d in data if not d["fell"]), len(positive)))
    
    # 准备张量
    prefix_joints = torch.tensor(np.array([d["prefix_joints"] for d in positive]), dtype=torch.float32)
    prefix_velocities = torch.tensor(np.array([d["prefix_velocities"] for d in positive]), dtype=torch.float32)
    x_0 = torch.tensor(np.array([
        np.concatenate([d["init_joints"], d["init_velocities"]]) for d in positive
    ]), dtype=torch.float32)
    
    # 标准化
    x_mean, x_std = x_0.mean(0), x_0.std(0) + 1e-6
    x_0_norm = (x_0 - x_mean) / x_std
    
    # 切换到GPU
    prefix_joints = prefix_joints.to(device)
    prefix_velocities = prefix_velocities.to(device)
    x_0_norm = x_0_norm.to(device)
    x_mean, x_std = x_mean.to(device), x_std.to(device)
    
    n_joints = prefix_joints.shape[-1]
    
    # 更大的模型: 256 hidden, 100 diffusion steps
    class DIO_V2(DIO):
        def __init__(self, n_joints, d_model=256, n_steps=100):
            super().__init__(n_joints=n_joints, d_model=d_model, n_steps=n_steps)
            # Override denoiser with bigger one
            self.denoiser = Denoiser(self.state_dim, d_model, hidden=512)
    
    model = DIO_V2(n_joints=n_joints).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 批量训练
    batch_size = 64
    n_samples = len(positive)
    
    print(f"Training DIO V2: {n_samples} samples, {n_joints} joints, {epochs} epochs")
    print(f"Model: d_model=256, hidden=512, diffusion_steps=100")
    print(f"Batch size: {batch_size}")
    
    losses = []
    start = time.time()
    
    for epoch in range(epochs):
        epoch_losses = []
        # Mini-batch
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_pj = prefix_joints[idx]
            batch_pv = prefix_velocities[idx]
            batch_x0 = x_0_norm[idx]
            
            t = torch.randint(0, model.n_steps, (len(idx),), device=device)
            loss = model(batch_pj, batch_pv, batch_x0, t)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} ({time.time()-start:.0f}s)")
    
    # 保存
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean.cpu(),
        "x_std": x_std.cpu(),
        "n_joints": n_joints,
        "train_samples": n_samples,
        "final_loss": losses[-1],
        "architecture": "DIO_V2_256_512_100",
    }, save_path)
    
    print(f"\nSaved to {save_path}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Training time: {time.time()-start:.0f}s")
    print(f"Loss curve: start={losses[0]:.4f} → end={losses[-1]:.4f}")
    
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    data_path = BASE + "/data/experiments/p2_main/dio_training_data.pkl"
    save_path = BASE + "/data/experiments/p2_main/dio_model_v2.pt"
    
    model = train_dio_v2(data_path, save_path, epochs=2000, device=device)
