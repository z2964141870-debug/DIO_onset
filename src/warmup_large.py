#!/usr/bin/env python3
"""
大规模 Warm-up 实验（严谨性修复版）

改进:
1. 多测试帧: f100/f200/f300/f400/f500 (不只 f200)
2. 双策略: 035 + M17
3. 同参考长度控制: warm-up 版和标准版用完全相同的参考
   - 标准: 从帧 X 开始, 300 帧, 统计全部摔倒
   - Warm-up: 从帧 X-50 开始, 350 帧, 只统计 >1.67s 后的摔倒
   - 两者看到的参考不同但逻辑正确
4. 更大样本: 10 sessions × 5 frames × 2 arms = 100 pairs
"""
import numpy as np
import os, subprocess, json, glob
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "/opt/data/private/yuanqi_ops"
PY = "/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python"
S2S = "/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py"
MJCF = "/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
CK035 = "/opt/data/private/hechengzhu/ZHY/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
CK17 = BASE + "/data/training/p2_main/m17/checkpoints/milestone_4/model_step_000004.pt"
V7 = BASE + "/data/gen_audit/sequences_v7"
WORK = BASE + "/data/experiments/p2_main/warmup_large"
os.makedirs(WORK, exist_ok=True)

WARMUP = 50  # frames
TEST_FRAMES = [100, 200, 300, 400, 500]

def run_one(args):
    sess_dir, test_frame, arm, ckpt = args
    sess = os.path.basename(sess_dir.rstrip("/"))
    short = sess[-6:]
    
    ref_csv = os.path.join(sess_dir, "reference.csv")
    if not os.path.exists(ref_csv):
        return None
    
    rows = np.loadtxt(ref_csv, delimiter=",", skiprows=1)
    if rows.ndim == 1: rows = rows.reshape(1, -1)
    with open(ref_csv, "r") as f:
        header = f.readline()
    
    results = {}
    
    # 条件A: 标准起始（从 test_frame 开始, 300帧, 全部摔倒算）
    a_dir = os.path.join(WORK, "%s_f%d_%s_std" % (short, test_frame, arm))
    a_mj = os.path.join(a_dir, "m.json")
    if not os.path.exists(a_mj):
        os.makedirs(a_dir, exist_ok=True)
        a_ref = os.path.join(a_dir, "ref.csv")
        s, e = test_frame, min(test_frame + 300, len(rows))
        with open(a_ref, "w") as f:
            f.write(header)
            for r in rows[s:e]:
                f.write(",".join(str(x) for x in r) + "\n")
        subprocess.run([PY, S2S, "--checkpoint", ckpt, "--motion", a_ref,
            "--encoder-mode", "a3_fast", "--csv-source-fps", "30",
            "--csv-frame-stride", "1", "--mjcf", MJCF, "--batch-once",
            "--action-delay-ms", "0", "--metrics-out", a_mj],
            capture_output=True, timeout=120)
    
    if os.path.exists(a_mj):
        j = json.load(open(a_mj))
        results["std_fell"] = bool(j.get("fall"))
        results["std_time"] = j.get("fall_time_s", -1)
    
    # 条件B: Warm-up（从 test_frame-50 开始, 350帧, 只算>1.67s 的摔倒）
    warm_start = test_frame - WARMUP
    if warm_start < 0:
        warm_start = 0
    b_dir = os.path.join(WORK, "%s_f%d_%s_warm" % (short, test_frame, arm))
    b_mj = os.path.join(b_dir, "m.json")
    if not os.path.exists(b_mj):
        os.makedirs(b_dir, exist_ok=True)
        b_ref = os.path.join(b_dir, "ref.csv")
        s, e = warm_start, min(warm_start + WARMUP + 300, len(rows))
        with open(b_ref, "w") as f:
            f.write(header)
            for r in rows[s:e]:
                f.write(",".join(str(x) for x in r) + "\n")
        subprocess.run([PY, S2S, "--checkpoint", ckpt, "--motion", b_ref,
            "--encoder-mode", "a3_fast", "--csv-source-fps", "30",
            "--csv-frame-stride", "1", "--mjcf", MJCF, "--batch-once",
            "--action-delay-ms", "0", "--metrics-out", b_mj],
            capture_output=True, timeout=120)
    
    if os.path.exists(b_mj):
        j = json.load(open(b_mj))
        b_fell_raw = bool(j.get("fall"))
        b_time = j.get("fall_time_s", -1)
        threshold = WARMUP / 30.0
        results["warm_fell_test"] = b_fell_raw and b_time > threshold
        results["warm_fell_during"] = b_fell_raw and b_time <= threshold
        results["warm_time"] = b_time
    
    results["sess"] = short
    results["frame"] = test_frame
    results["arm"] = arm
    return results

# 生成任务
tasks = []
sessions = sorted(glob.glob(V7 + "/*/"))
for sess_dir in sessions:
    for tf in TEST_FRAMES:
        for arm, ck in [("035", CK035), ("m17", CK17)]:
            tasks.append((sess_dir, tf, arm, ck))

print("Total evaluations: %d (10 sessions x 5 frames x 2 arms x 2 conditions)" % len(tasks))

# 4 路并行
results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_one, t): t for t in tasks}
    for i, future in enumerate(as_completed(futures)):
        r = future.result()
        if r:
            results.append(r)
        if (i+1) % 20 == 0:
            print("  %d/%d done" % (i+1, len(tasks)))

# 分析
print("\n" + "=" * 70)
print("LARGE-SCALE WARM-UP RESULTS")
print("=" * 70)

for arm in ["035", "m17"]:
    arm_results = [r for r in results if r["arm"] == arm]
    print("\nPolicy: %s" % arm)
    
    std_falls = sum(1 for r in arm_results if r.get("std_fell"))
    warm_test_falls = sum(1 for r in arm_results if r.get("warm_fell_test"))
    warm_during_falls = sum(1 for r in arm_results if r.get("warm_fell_during"))
    total_falls_warm = warm_test_falls + warm_during_falls
    
    n = len(arm_results)
    print("  n_pairs: %d" % n)
    print("  Standard falls: %d/%d (%.0f%%)" % (std_falls, n, std_falls/n*100 if n else 0))
    print("  Warm-up falls (test window): %d/%d (%.0f%%)" % (warm_test_falls, n, warm_test_falls/n*100 if n else 0))
    print("  Warm-up falls (during warm-up): %d/%d" % (warm_during_falls, n))
    print("  Warm-up total falls: %d/%d" % (total_falls_warm, n))
    
    # McNemar
    b_count = sum(1 for r in arm_results if r.get("std_fell") and not r.get("warm_fell_test"))
    c_count = sum(1 for r in arm_results if not r.get("std_fell") and r.get("warm_fell_test"))
    print("  McNemar: std_fall+warm_OK=%d, std_OK+warm_fall=%d" % (b_count, c_count))
    
    if b_count + c_count > 0:
        from scipy.stats import binomtest
        p = binomtest(b_count, b_count + c_count, 0.5).pvalue
        print("  Exact p = %.4f" % p)
        if p < 0.05:
            print("  ✅ SIGNIFICANT")
        else:
            print("  ⚠️ Not significant")

# 保存全部结果
import pickle
with open(WORK + "/results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nResults saved to %s/results.pkl" % WORK)
