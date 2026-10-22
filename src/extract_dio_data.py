#!/usr/bin/env python3
"""
DIO 训练数据提取
从 390 个评估中提取:
- condition: 参考前 30 帧 (joint angles + velocities)
- initial_state: 起始帧的关节角 + 速度 (成功 vs 失败)
- outcome: fall / no_fall
"""
import json, glob, os, re
import numpy as np
import pickle

BASE = "/opt/data/private/yuanqi_ops"

def load_all_data():
    """加载去重后的所有评估 + 对应的参考特征"""
    seen = {}
    for exp_dir in ["onset_large", "onset_xfine"]:
        base = f"{BASE}/data/experiments/p2_main/{exp_dir}"
        for p in sorted(glob.glob(f"{base}/*.metrics.json")):
            fname = os.path.basename(p).replace(".metrics.json", "")
            parts = fname.split("__")
            session_part = parts[0]
            arm = parts[1]
            if "_f" not in session_part:
                continue
            sess = session_part.rsplit("_f", 1)[0]
            start_str = session_part.rsplit("_f", 1)[1]
            try:
                start = int(start_str)
            except:
                continue
            key = (sess, start, arm)
            if key in seen:
                continue
            
            j = json.load(open(p))
            fell = bool(j.get("fall"))
            ori = (j.get("anchor_pose") or {}).get("orientation_max_deg") or 0
            
            # 找参考文件
            ref = None
            for stage in ["onset_large", "onset_xfine"]:
                rp = f"{BASE}/data/gen_audit/{stage}/{session_part}/reference.csv"
                if os.path.exists(rp):
                    ref = rp
                    break
            if not ref:
                continue
            
            seen[key] = dict(
                sess=sess, start=start, arm=arm,
                fell=fell, ori=ori, ref_path=ref
            )
    return list(seen.values())

def extract_reference_features(ref_path, prefix_len=30):
    """
    从参考 CSV 提取特征
    列结构: [0]=Frame, [1:4]=root_trans_xyz, [4:7]=root_rot_xyz, [7:]=joint_angles
    
    返回:
    - prefix_joints: (prefix_len, n_joints) 参考前缀关节角
    - prefix_velocities: (prefix_len, n_joints) 参考前缀关节速度(差分)
    - init_joints: (n_joints,) 起始帧关节角
    - init_velocities: (n_joints,) 起始帧关节速度(从差分计算)
    """
    rows = np.loadtxt(ref_path, delimiter=",", skiprows=1)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    
    T = len(rows)
    joints = rows[:, 7:]  # (T, n_joints)
    n_joints = joints.shape[1]
    
    # 关节速度 (差分, 30fps)
    if T > 1:
        velocities = np.zeros_like(joints)
        velocities[1:] = (joints[1:] - joints[:-1]) * 30.0  # per second
        velocities[0] = velocities[1] if T > 1 else 0
    else:
        velocities = np.zeros_like(joints)
    
    # 前缀
    n = min(prefix_len, T)
    prefix_joints = joints[:n]
    prefix_velocities = velocities[:n]
    
    # 如果前缀不足，用 0 填充
    if n < prefix_len:
        pad_j = np.zeros((prefix_len - n, n_joints))
        pad_v = np.zeros((prefix_len - n, n_joints))
        prefix_joints = np.vstack([prefix_joints, pad_j])
        prefix_velocities = np.vstack([prefix_velocities, pad_v])
    
    # 起始状态
    init_joints = joints[0]
    init_velocities = velocities[0]
    
    return {
        "prefix_joints": prefix_joints,
        "prefix_velocities": prefix_velocities,
        "init_joints": init_joints,
        "init_velocities": init_velocities,
        "n_joints": n_joints,
        "T": T,
    }

def main():
    evals = load_all_data()
    print(f"Loaded {len(evals)} unique evaluations")
    
    # 只用 035 策略的数据（避免策略间差异干扰训练）
    evals_035 = [e for e in evals if e["arm"] == "035"]
    print(f"Policy 035: {len(evals_035)} evaluations")
    print(f"  Falls: {sum(1 for e in evals_035 if e['fell'])}")
    print(f"  Survives: {sum(1 for e in evals_035 if not e['fell'])}")
    
    # 提取特征
    data = []
    for e in evals_035:
        feats = extract_reference_features(e["ref_path"])
        data.append({
            "sess": e["sess"],
            "start": e["start"],
            "fell": e["fell"],
            "ori": e["ori"],
            **feats
        })
    
    # 统计
    n_joints = data[0]["n_joints"]
    print(f"\nJoints per frame: {n_joints}")
    print(f"Prefix length: {data[0]['prefix_joints'].shape}")
    print(f"Init state dim: {data[0]['init_joints'].shape[0]} (joints) + {data[0]['init_velocities'].shape[0]} (velocities)")
    
    # 分析成功 vs 失败的初始状态差异
    falls = [d for d in data if d["fell"]]
    survs = [d for d in data if not d["fell"]]
    
    print(f"\n=== Success vs Failure initial state comparison ===")
    for name in ["init_joints", "init_velocities"]:
        f_val = np.array([d[name] for d in falls])
        s_val = np.array([d[name] for d in survs])
        print(f"  {name}:")
        print(f"    Fall:    mean={f_val.mean():.3f} std={f_val.std():.3f}")
        print(f"    Survive: mean={s_val.mean():.3f} std={s_val.std():.3f}")
        
        # 逐关节差异
        diff = np.abs(f_val.mean(0) - s_val.mean(0))
        top5 = np.argsort(diff)[-5:]
        print(f"    Top-5 discriminative joints: {top5} (diff={diff[top5]:.3f})")
    
    # 保存训练数据
    outpath = f"{BASE}/data/experiments/p2_main/dio_training_data.pkl"
    with open(outpath, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved {len(data)} samples to {outpath}")
    
    # 保存摘要统计
    print(f"\n=== Data Summary for DIO Training ===")
    print(f"Total samples: {len(data)}")
    print(f"Positive (survive): {len(survs)}")
    print(f"Negative (fall): {len(falls)}")
    print(f"Condition dim: {30 * n_joints * 2} (prefix joints + velocities)")
    print(f"Output dim: {n_joints * 2} (init joints + velocities)")

if __name__ == "__main__":
    main()
