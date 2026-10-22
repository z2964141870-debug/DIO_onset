#!/usr/bin/env python3
import json, glob, os, csv
import numpy as np
from collections import defaultdict

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0, 0, 1
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return p, max(0, center - margin), min(1, center + margin)

def compute_features(ref_path):
    rows = np.loadtxt(ref_path, delimiter=",", skiprows=1)
    if rows.ndim == 1: rows = rows.reshape(1, -1)
    T = len(rows)
    n = min(30, T - 1)
    dt = n / 30.0
    root_z = rows[0, 3]
    joints = rows[:, 7:]
    jspeed = np.abs(joints[n] - joints[0]).mean() / dt if n > 0 and dt > 0 else 0
    jspeed5 = np.abs(joints[min(5,n)] - joints[0]).mean() / (min(5,n)/30.0) if n > 0 else 0
    rdisp = np.linalg.norm(rows[n, 1:4] - rows[0, 1:4]) if n > 0 else 0
    pmag = np.abs(joints[0]).mean()
    pvar = np.var(joints[0])
    return dict(root_z=root_z, jspeed=jspeed, jspeed5=jspeed5,
                rdisp=rdisp, pmag=pmag, pvar=pvar)

# 加载 + 去重(字符串解析,不用正则)
data = []
seen = set()
for exp_dir in ["onset_large", "onset_xfine"]:
    pattern = "/opt/data/private/yuanqi_ops/data/experiments/p2_main/" + exp_dir + "/*.metrics.json"
    for p in sorted(glob.glob(pattern)):
        fname = os.path.basename(p).replace(".metrics.json", "")
        parts = fname.split("__")
        session_part = parts[0]
        arm = parts[1]
        if "_f" not in session_part: continue
        sess = session_part.rsplit("_f", 1)[0]
        start_str = session_part.rsplit("_f", 1)[1]
        try: start = int(start_str)
        except: continue
        key = (sess, start, arm)
        if key in seen: continue
        seen.add(key)
        j = json.load(open(p))
        fell = bool(j.get("fall"))
        ori = (j.get("anchor_pose") or {}).get("orientation_max_deg") or 0
        ref = None
        for stage in ["onset_large", "onset_xfine"]:
            rp = "/opt/data/private/yuanqi_ops/data/gen_audit/" + stage + "/" + session_part + "/reference.csv"
            if os.path.exists(rp): ref = rp; break
        if not ref: continue
        feats = compute_features(ref)
        data.append(dict(sess=sess[-6:], start=start, arm=arm, fell=fell, ori=ori, **feats))

print("Loaded %d unique evaluations" % len(data))
print("Policies: %s" % sorted(set(d["arm"] for d in data)))
print("Sessions: %s" % sorted(set(d["sess"] for d in data)))

# 1. 摔倒率 Wilson CI
print()
print("=" * 60)
print("1. FALL RATE (Wilson 95% CI, deduplicated)")
print("=" * 60)
for arm in ["035", "m17"]:
    d = [x for x in data if x["arm"] == arm]
    print()
    print("Policy: %s" % arm)
    for st in sorted(set(x["start"] for x in d)):
        cell = [x for x in d if x["start"] == st]
        n = len(cell)
        k = sum(1 for x in cell if x["fell"])
        p, lo, hi = wilson_ci(k, n)
        mo = np.mean([x["ori"] for x in cell])
        print("  f%3d: %2d/%2d (%5.1f%%)  CI=[%4.1f%%,%5.1f%%]  ori=%6.1f" % (st, k, n, p*100, lo*100, hi*100, mo))

# 2. McNemar
print()
print("=" * 60)
print("2. McNEMAR (f0 vs f200, 035)")
print("=" * 60)
try:
    from scipy.stats import binomtest
except: binomtest = None
d035 = [x for x in data if x["arm"] == "035"]
sessions = sorted(set(x["sess"] for x in d035))
b = c = 0
for s in sessions:
    f0 = next((x for x in d035 if x["sess"] == s and x["start"] == 0), None)
    f200 = next((x for x in d035 if x["sess"] == s and x["start"] == 200), None)
    if f0 and f200:
        if f0["fell"] and not f200["fell"]: b += 1; print("  %s: f0 FALL -> f200 OK" % s)
        elif not f0["fell"] and f200["fell"]: c += 1; print("  %s: f0 OK -> f200 FALL" % s)
print("  b=%d c=%d" % (b, c))
if b + c > 0 and binomtest:
    pv = binomtest(b, b+c, 0.5).pvalue
    print("  Exact p = %.4f" % pv)

# 3. 预测 GroupKFold
print()
print("=" * 60)
print("3. PREDICTION (GroupKFold by session)")
print("=" * 60)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GroupKFold
feats = ["root_z", "jspeed", "jspeed5", "rdisp", "pmag", "pvar"]
X = np.array([[x[f] for f in feats] for x in d035])
y = np.array([x["fell"] for x in d035])
groups = np.array([x["sess"] for x in d035])
print("  n=%d fall_rate=%.1f%% sessions=%d" % (len(y), y.mean()*100, len(set(groups))))
rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=0, max_depth=4))
gkf = GroupKFold(n_splits=min(5, len(set(groups))))
scores = cross_val_score(rf, X, y, cv=gkf, groups=groups, scoring="accuracy")
print("  RF GroupKFold: %.1f%% +/- %.1f%%" % (scores.mean()*100, scores.std()*100))
print("  Baseline: %.1f%%" % (y.mean()*100))
rf.fit(X, y)
imp = sorted(zip(feats, rf.named_steps["randomforestclassifier"].feature_importances_), key=lambda t: -t[1])
print("  Feature importance:")
for f, v in imp: print("    %10s: %.3f" % (f, v))

# 4. 导出
outpath = "/opt/data/private/yuanqi_ops/reports/onset_corrected.csv"
fields = ["sess", "start", "arm", "fell", "ori"] + feats
with open(outpath, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row in data: w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
print()
print("Exported %d rows to %s" % (len(data), outpath))
