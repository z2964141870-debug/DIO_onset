#!/usr/bin/env python3
"""
DIO: Diffusion Initialization Optimizer
用于人形跟踪策略的最优初始化状态生成

架构:
  条件编码器: 参考前缀 (30×31×2) → Transformer → 128维
  扩散去噪器: (62维状态 + 128维条件 + 时间步) → MLP → 62维
  
训练: 去噪分数匹配 (DDPM)
推理: 采样 K 个候选 → 选最优
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import time

class ConditionEncoder(nn.Module):
    """编码参考前缀 → 固定长度条件向量"""
    def __init__(self, n_joints=31, prefix_len=30, d_model=128):
        super().__init__()
        self.n_joints = n_joints
        self.prefix_len = prefix_len
        input_dim = n_joints * 2  # joints + velocities
        
        self.frame_embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(prefix_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, prefix_joints, prefix_velocities):
        """
        prefix_joints: (B, T, J)
        prefix_velocities: (B, T, J)
        Returns: (B, d_model)
        """
        B, T, J = prefix_joints.shape
        x = torch.cat([prefix_joints, prefix_velocities], dim=-1)  # (B, T, 2J)
        x = self.frame_embed(x)  # (B, T, D)
        x = x + self.pos_embed[:T].unsqueeze(0)  # add positional encoding
        x = self.transformer(x)  # (B, T, D)
        x = x.mean(dim=1)  # (B, D) - mean pooling
        return x


class Denoiser(nn.Module):
    """扩散去噪网络: 预测噪声 ε"""
    def __init__(self, state_dim=62, cond_dim=128, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        # 主网络
        input_dim = state_dim + cond_dim + 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        
    def forward(self, x_t, t, cond):
        """
        x_t: (B, state_dim) 加噪状态
        t: (B,) 时间步
        cond: (B, cond_dim) 条件
        Returns: (B, state_dim) 预测的噪声
        """
        t_embed = self.time_embed(t.float().unsqueeze(-1))  # (B, 64)
        inp = torch.cat([x_t, cond, t_embed], dim=-1)
        return self.net(inp)


class DIO(nn.Module):
    """完整的 DIO 模型"""
    def __init__(self, n_joints=31, prefix_len=30, d_model=128, n_steps=50):
        super().__init__()
        self.n_joints = n_joints
        self.state_dim = n_joints * 2  # joints + velocities
        self.n_steps = n_steps
        
        self.encoder = ConditionEncoder(n_joints, prefix_len, d_model)
        self.denoiser = Denoiser(self.state_dim, d_model)
        
        # DDPM 噪声调度
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, 0))
        
    def forward(self, prefix_joints, prefix_velocities, x_0, t):
        """训练前向: 加噪 + 预测噪声"""
        cond = self.encoder(prefix_joints, prefix_velocities)
        
        # DDPM 前向过程
        noise = torch.randn_like(x_0)
        alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        
        # 预测噪声
        noise_pred = self.denoiser(x_t, t, cond)
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def sample(self, prefix_joints, prefix_velocities, n_samples=10):
        """推理: 采样多个候选初始化"""
        cond = self.encoder(prefix_joints, prefix_velocities)
        B = prefix_joints.shape[0]
        
        # 从纯噪声开始
        x = torch.randn(B * n_samples, self.state_dim, device=prefix_joints.device)
        cond_rep = cond.repeat_interleave(n_samples, dim=0)
        
        # 反向去噪
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((B * n_samples,), t, device=x.device, dtype=torch.long)
            noise_pred = self.denoiser(x, t_batch, cond_rep)
            
            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            
            if t > 0:
                noise = torch.randn_like(x)
                x = (1 / torch.sqrt(alpha)) * (
                    x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred
                ) + torch.sqrt(self.betas[t]) * noise
            else:
                x = (1 / torch.sqrt(alpha)) * (
                    x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred
                )
        
        return x.view(B, n_samples, self.state_dim)


def train_dio(data_path, save_path, epochs=200, lr=1e-3, device="cuda"):
    """训练 DIO 模型"""
    print(f"Loading data from {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # 只用成功样本训练（生成"好的初始化"）
    positive = [d for d in data if not d["fell"]]
    print(f"Positive samples (survive): {len(positive)}")
    
    if len(positive) < 10:
        print("ERROR: Too few positive samples")
        return None
    
    # 准备张量
    prefix_joints = torch.tensor(
        np.array([d["prefix_joints"] for d in positive]), dtype=torch.float32
    )
    prefix_velocities = torch.tensor(
        np.array([d["prefix_velocities"] for d in positive]), dtype=torch.float32
    )
    # 目标状态: 初始关节角 + 速度
    x_0 = torch.tensor(
        np.array([
            np.concatenate([d["init_joints"], d["init_velocities"]]) for d in positive
        ]), dtype=torch.float32
    )
    
    # 标准化
    x_mean, x_std = x_0.mean(0), x_0.std(0) + 1e-6
    x_0_norm = (x_0 - x_mean) / x_std
    
    prefix_joints = prefix_joints.to(device)
    prefix_velocities = prefix_velocities.to(device)
    x_0_norm = x_0_norm.to(device)
    x_mean, x_std = x_mean.to(device), x_std.to(device)
    
    n_joints = prefix_joints.shape[-1]
    model = DIO(n_joints=n_joints).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Training DIO: {len(positive)} samples, {n_joints} joints, {epochs} epochs")
    print(f"State dim: {model.state_dim}, Diffusion steps: {model.n_steps}")
    
    losses = []
    start = time.time()
    
    for epoch in range(epochs):
        # 随机时间步
        t = torch.randint(0, model.n_steps, (len(positive),), device=device)
        
        # 随机打乱
        perm = torch.randperm(len(positive))
        loss = model(
            prefix_joints[perm], prefix_velocities[perm], x_0_norm[perm], t[perm]
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f} ({time.time()-start:.0f}s)")
    
    # 保存
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean.cpu(),
        "x_std": x_std.cpu(),
        "n_joints": n_joints,
        "train_samples": len(positive),
        "final_loss": losses[-1],
    }, save_path)
    print(f"\nSaved model to {save_path}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Training time: {time.time()-start:.0f}s")
    
    return model


if __name__ == "__main__":
    BASE = "/opt/data/private/yuanqi_ops"
    data_path = f"{BASE}/data/experiments/p2_main/dio_training_data.pkl"
    save_path = f"{BASE}/data/experiments/p2_main/dio_model.pt"
    
    # 检测GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = train_dio(data_path, save_path, epochs=200, device=device)
    
    if model is not None:
        # 快速测试: 采样并检查输出
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        test = data[0]
        pj = torch.tensor(test["prefix_joints"], dtype=torch.float32).unsqueeze(0).to(device)
        pv = torch.tensor(test["prefix_velocities"], dtype=torch.float32).unsqueeze(0).to(device)
        
        samples = model.sample(pj, pv, n_samples=5)
        ckpt = torch.load(save_path, map_location=device)
        samples_denorm = samples * ckpt["x_std"] + ckpt["x_mean"]
        
        print(f"\n=== Sample Generation Test ===")
        print(f"Input prefix shape: {pj.shape}")
        print(f"Generated samples shape: {samples.shape}")
        print(f"Sample 0 joints (first 5): {samples_denorm[0, 0, :5].cpu().detach().numpy()}")
        print(f"Sample 0 velocities (first 5): {samples_denorm[0, 0, 31:36].cpu().detach().numpy()}")
        print(f"\nDIO training complete!")
