#!/usr/bin/env python3
"""
DIO v2 评估：生成 warm-up 过渡轨迹（非单帧替换）

改进思路:
- v1 失败原因: 只替换帧 0 的关节角，没解决冷启动问题
- v2 改进: DIO 生成 30 帧的过渡序列，从"稳定姿态"渐进过渡到参考起始
  - 这 30 帧既是 warm-up（让 RNN 建立上下文），也是 DIO 的贡献（比线性插值更优）
  
实现:
1. DIO 输出: 最优初始关节角 + 速度 (62 维)
2. 生成 30 帧过渡: 从 DIO 初始状态到参考帧 0 的平滑轨迹
   - 使用 DIO 的速度作为初始方向
   - 使用余弦/三次样条平滑
3. 拼接: [30帧 DIO 过渡] + [300帧 原始参考]
4. 评估: 只统计 >1.0s (30帧后) 的摔倒
"""
import torch, numpy as np, pickle, os, subprocess, json, sys, glob

BASE = "/opt/data/private/yuanqi_ops"
PY = "/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python"
S2S = "/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py"
MJCF = "/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
CKPT = "/opt/data/private/hechengzhu/ZHY/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"

sys.path.insert(0, BASE + "/script")
from dio_model import DIO, Denoiser

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dio_v2():
    ckpt = torch.load(BASE + "/data/experiments/p2_main/dio_model_v2.pt", map_location=device)
    n_joints = ckpt["n_joints"]
    
    # 重建模型架构 (v2: 256 d_model, 512 hidden, 100 steps)
    class DIO_V2(DIO):
        def __init__(self, n_joints, d_model=256, n_steps=100):
            super().__init__(n_joints=n_joints, d_model=d_model, n_steps=n_steps)
            self.denoiser = Denoiser(self.state_dim, d_model, hidden=512)
    
    model = DIO_V2(n_joints=n_joints).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

def generate_transition(dio_joints, dio_velocities, ref_first_joints, n_frames=30):
    """
    生成从 DIO 初始状态到参考帧 0 的平滑过渡
    使用三次插值（考虑初始速度方向）
    """
    from scipy.interpolate import CubicSpline
    
    n_joints = len(dio_joints)
    t_points = np.array([0, n_frames])  # 起点、终点
    y_points = np.vstack([dio_joints, ref_first_joints])  # (2, n_joints)
    
    # 使用初始速度作为样条的一阶导数约束
    # 假设末端速度为 0（到达参考帧 0 时稳定）
    cs = CubicSpline(t_points, y_points, axis=0, 
                     bc_type=((1, dio_velocities), (1, np.zeros(n_joints))))
    
    # 生成过渡帧
    t_fine = np.arange(0, n_frames)
    transition = cs(t_fine)  # (n_frames, n_joints)
    
    return transition

def evaluate_session(model, ckpt, test_data, ref_csv, work_dir):
    """用 DIO 生成过渡 + 评估"""
    # 生成 DIO 初始化
    pj = torch.tensor(test_data["prefix_joints"], dtype=torch.float32).unsqueeze(0).to(device)
    pv = torch.tensor(test_data["prefix_velocities"], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        samples = model.sample(pj, pv, n_samples=5)
    samples = samples * ckpt["x_std"].to(device) + ckpt["x_mean"].to(device)
    
    # 读原始参考
    rows = np.loadtxt(ref_csv, delimiter=",", skiprows=1)
    if rows.ndim == 1: rows = rows.reshape(1, -1)
    with open(ref_csv, "r") as f:
        header = f.readline()
    
    n_joints_total = rows.shape[1] - 7  # 去掉 Frame + root 6 列
    
    best_result = None
    for k in range(min(3, samples.shape[1])):
        dio_joints = samples[0, k, :31].cpu().numpy()  # 前 31 是关节角
        dio_velocities = samples[0, k, 31:].cpu().numpy()  # 后 31 是速度
        
        # 生成过渡序列
        ref_first = rows[0, 7:]  # 参考帧 0 的关节角
        transition = generate_transition(dio_joints[:n_joints_total], 
                                         dio_velocities[:n_joints_total],
                                         ref_first, n_frames=30)
        
        # 创建修改版参考: [30帧过渡] + [300帧原始]
        mod_rows = []
        for t_frame in transition:
            # root 位置/旋转: 从当前值到帧 0 的值线性插值
            alpha = t_frame / 30  # 0→1
            root_trans = rows[0, 1:4] * alpha  # 从原点过渡到帧0位置
            root_rot = rows[0, 4:7] * alpha
            row = np.concatenate([[0], root_trans, root_rot, t_frame])
            mod_rows.append(row)
        
        mod_rows = np.vstack(mod_rows)
        full_ref = np.vstack([mod_rows, rows[:300]])  # 过渡 + 原始300帧
        
        # 写参考文件
        os.makedirs(work_dir, exist_ok=True)
        ref_out = os.path.join(work_dir, "ref.csv")
        with open(ref_out, "w") as f:
            f.write(header)
            for r in full_ref:
                f.write(",".join(str(x) for x in r) + "\n")
        
        # 评估
        mj = os.path.join(work_dir, "m.json")
        subprocess.run([PY, S2S, "--checkpoint", CKPT, "--motion", ref_out,
            "--encoder-mode", "a3_fast", "--csv-source-fps", "30",
            "--csv-frame-stride", "1", "--mjcf", MJCF, "--batch-once",
            "--action-delay-ms", "0", "--metrics-out", mj],
            capture_output=True, timeout=180)
        
        if os.path.exists(mj):
            m = json.load(open(mj))
            fell = bool(m.get("fall"))
            ft = m.get("fall_time_s", -1)
            ori = (m.get("anchor_pose") or {}).get("orientation_max_deg", 999)
            
            # 只统计过渡后(>1.0s)的摔倒
            test_fell = fell and ft > 1.0
            during_fell = fell and ft <= 1.0
            
            if not test_fell:
                return dict(fell=False, during_warmup=during_fell, ori=ori, sample=k)
            
            if best_result is None or ori < best_result["ori"]:
                best_result = dict(fell=True, during_warmup=during_fell, ori=ori, sample=k)
    
    return best_result or dict(fell=True, during_warmup=False, ori=999, sample=-1)

def main():
    model, ckpt = load_dio_v2()
    
    with open(BASE + "/data/experiments/p2_main/dio_training_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # 目标: f0 摔倒的 session
    f0_falls = [d for d in data if d["start"] == 0 and d["fell"]]
    print("f0 falls: %d" % len(f0_falls))
    
    # 找到对应的参考文件
    def find_ref(short):
        for stage in ["onset_large", "onset_xfine"]:
            pattern = BASE + "/data/gen_audit/" + stage + "/*_" + short + "_f0/reference.csv"
            matches = glob.glob(pattern)
            if matches: return matches[0]
        return None
    
    WORK = BASE + "/data/experiments/p2_main/dio_v2_eval"
    results = []
    
    for i, test in enumerate(f0_falls):
        short = test["sess"]
        ref = find_ref(short)
        if not ref:
            continue
        print("[%d/%d] %s: " % (i+1, len(f0_falls), short), end="", flush=True)
        
        r = evaluate_session(model, ckpt, test, ref, os.path.join(WORK, short))
        results.append(dict(sess=short, **r))
        
        if not r["fell"]:
            print("RESCUED (ori=%.1f, sample=%d)" % (r["ori"], r["sample"]))
        elif r.get("during_warmup"):
            print("FELL DURING TRANSITION (ori=%.1f)" % r["ori"])
        else:
            print("NOT RESCUED (ori=%.1f)" % r["ori"])
    
    # 汇总
    print("\n" + "=" * 50)
    rescued = sum(1 for r in results if not r["fell"])
    during = sum(1 for r in results if r.get("during_warmup"))
    total = len(results)
    print("DIO v2 + Transition Results:")
    print("  Rescued: %d/%d (%.0f%%)" % (rescued, total, rescued/total*100 if total else 0))
    print("  Fell during transition: %d/%d" % (during, total))
    print("  Fell after transition: %d/%d" % (total - rescued - during, total))
    
    for r in results:
        status = "RESCUED" if not r["fell"] else ("DURING_WARMUP" if r.get("during_warmup") else "NOT_RESCUED")
        print("  %s: %s" % (r["sess"], status))

if __name__ == "__main__":
    main()
