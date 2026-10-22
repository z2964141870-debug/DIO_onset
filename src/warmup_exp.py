#!/usr/bin/env python3
"""
Warm-up 实验: 前滚 50 帧建立 RNN 上下文，然后评估正式窗口

设计:
- 标准 f200 评估: 机器人在 f200 初始化(冷启动)，跟踪 f200-f500
- Warm-up 评估: 机器人在 f150 初始化，先跑 f150-f200 (warm-up)，
  然后继续跟踪 f200-f500 (正式窗口)
- 如果 warm-up 版在正式窗口内摔倒率更低 → 冷启动假设成立

实现: 参考从 f150 开始(350帧)，评估整个 episode，
只统计 fall_time_s > 50/30=1.67s 的摔倒(即正式窗口内的摔倒)
"""
import numpy as np
import os, subprocess, json, glob

BASE = "/opt/data/private/yuanqi_ops"
PY = "/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python"
S2S = "/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py"
MJCF = "/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
CKPT = "/opt/data/private/hechengzhu/ZHY/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
V7 = BASE + "/data/gen_audit/sequences_v7"

WORK = BASE + "/data/experiments/p2_main/warmup_exp"
os.makedirs(WORK, exist_ok=True)

WARMUP_FRAMES = 50  # 50帧 = 1.67秒
TEST_WINDOW = 300   # 正式评估窗口

def run_eval(ref_path, out_json):
    cmd = [PY, S2S, "--checkpoint", CKPT, "--motion", ref_path,
           "--encoder-mode", "a3_fast", "--csv-source-fps", "30",
           "--csv-frame-stride", "1", "--mjcf", MJCF, "--batch-once",
           "--action-delay-ms", "0", "--metrics-out", out_json]
    subprocess.run(cmd, capture_output=True, timeout=180)
    if os.path.exists(out_json):
        return json.load(open(out_json))
    return None

# 对每个 v7 session，测试 f200 起始（标准 vs warm-up）
results = []
sessions = sorted(glob.glob(V7 + "/*/"))

for sess_dir in sessions:
    sess = os.path.basename(sess_dir.rstrip("/"))
    ref_csv = os.path.join(sess_dir, "reference.csv")
    if not os.path.exists(ref_csv):
        continue
    short = sess[-6:]
    
    rows = np.loadtxt(ref_csv, delimiter=",", skiprows=1)
    if rows.ndim == 1: rows = rows.reshape(1, -1)
    with open(ref_csv, "r") as f:
        header = f.readline()
    
    # 条件A: 标准 f200 (从帧200开始，300帧)
    # 机器人冷启动在 f200
    a_dir = os.path.join(WORK, short + "_std")
    os.makedirs(a_dir, exist_ok=True)
    a_ref = os.path.join(a_dir, "ref.csv")
    with open(a_ref, "w") as f:
        f.write(header)
        for r in rows[200:501]:
            f.write(",".join(str(x) for x in r) + "\n")
    a_m = os.path.join(a_dir, "m.json")
    a_res = run_eval(a_ref, a_m)
    
    # 条件B: Warm-up f150→f500 (从帧150开始，350帧)
    # 机器人冷启动在 f150，前50帧是warm-up，f200后是正式窗口
    b_dir = os.path.join(WORK, short + "_warm")
    os.makedirs(b_dir, exist_ok=True)
    b_ref = os.path.join(b_dir, "ref.csv")
    with open(b_ref, "w") as f:
        f.write(header)
        for r in rows[150:501]:  # 350 帧
            f.write(",".join(str(x) for x in r) + "\n")
    b_m = os.path.join(b_dir, "m.json")
    b_res = run_eval(b_ref, b_m)
    
    # 分析结果
    a_fell = bool(a_res.get("fall")) if a_res else None
    a_time = a_res.get("fall_time_s", -1) if a_res else -1
    
    b_fell_raw = bool(b_res.get("fall")) if b_res else None
    b_time = b_res.get("fall_time_s", -1) if b_res else -1
    
    # Warm-up 版: 只统计正式窗口内(>1.67s)的摔倒
    warmup_threshold = WARMUP_FRAMES / 30.0  # 1.67s
    b_fell_test = b_fell_raw and b_time > warmup_threshold if b_fell_raw is not None else None
    b_fell_warmup = b_fell_raw and b_time <= warmup_threshold if b_fell_raw is not None else None
    
    results.append(dict(
        sess=short,
        a_fell=a_fell, a_time=a_time,
        b_fell_test=b_fell_test, b_fell_warmup=b_fell_warmup, b_time=b_time
    ))
    
    a_s = "FALL(t=%.1fs)" % a_time if a_fell else "OK"
    b_s = "WARMUP_FALL" if b_fell_warmup else ("TEST_FALL(t=%.1fs)" % b_time if b_fell_test else "OK")
    print("%s: std_f200=%s | warmup=%s" % (short, a_s, b_s))

# 汇总
print("\n" + "=" * 60)
print("WARM-UP EXPERIMENT RESULTS")
print("=" * 60)
n = len(results)
a_falls = sum(1 for r in results if r["a_fell"])
b_test_falls = sum(1 for r in results if r["b_fell_test"])
b_warmup_falls = sum(1 for r in results if r["b_fell_warmup"])
b_ok = sum(1 for r in results if r["b_fell_test"] is False)

print("Standard f200 (cold start):     %d/%d falls (%.0f%%)" % (a_falls, n, a_falls/n*100 if n else 0))
print("Warm-up f150→f200→f500:")
print("  Falls during warm-up (<1.67s): %d/%d" % (b_warmup_falls, n))
print("  Falls in test window (>1.67s): %d/%d (%.0f%%)" % (b_test_falls, n, b_test_falls/n*100 if n else 0))
print("  Survived entire episode:       %d/%d" % (b_ok, n))
print()
if b_test_falls < a_falls:
    print("✅ WARM-UP HELPS: %d → %d falls in test window" % (a_falls, b_test_falls))
else:
    print("❌ WARM-UP DOES NOT HELP or makes it worse")

# McNemar
a_and_b = sum(1 for r in results if r["a_fell"] and not r["b_fell_test"])
not_a_but_b = sum(1 for r in results if not r["a_fell"] and r["b_fell_test"])
print("  McNemar: A fall B OK = %d, A OK B fall = %d" % (a_and_b, not_a_but_b))
