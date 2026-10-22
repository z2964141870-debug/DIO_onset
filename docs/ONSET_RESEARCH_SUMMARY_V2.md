# Initialization Sensitivity in Humanoid Motion Tracking: An Exploratory Study

**Version**: 2.0 (corrected analysis)
**Date**: 2026-10-22
**Status**: Exploratory — descriptive findings with wide confidence intervals; mechanism not yet isolated

## Abstract

We observe that the reported fall rate of a humanoid tracking policy varies substantially depending on which frame of the reference motion the robot is initialized at. On 10 daily-motion sessions (wearable IMU, GT-aligned), the A3 tracking policy falls in 80% of episodes when initialized at the reference first frame (Wilson 95% CI: 49-94%), versus 40% at mid-motion frames (CI: 17-69%). A McNemar test on paired sessions shows a consistent direction (4 discordant pairs, all favoring mid-motion start) but does not reach significance (p=0.125, n=10). A random forest classifier using first-frame joint features achieves 67.5% accuracy in session-grouped cross-validation (baseline: 56.7%), providing weak but non-trivial predictive signal. We discuss the implications for evaluation protocol design and propose initialization-aware mitigation strategies.

## 1. Introduction

Standard evaluation protocols for humanoid tracking policies typically initialize the robot at the first frame of the reference motion. This convention implicitly assumes that the policy's performance is independent of where in the motion it begins tracking. We present evidence that this assumption may not hold for daily-motion tracking on a fine-tuned A3 humanoid.

**Key question**: Does the choice of initialization frame significantly affect measured tracking performance?

**Honest answer**: The descriptive evidence is suggestive but not conclusive. The effect direction is consistent across sessions, but the sample size (n=10) is insufficient for statistical significance, and content difficulty varies with start position.

## 2. Method

### 2.1 Setup
- **Robot**: AgiBot A3 (29 DOF)
- **Policies**: 035 official (primary), M17 fine-tuned (secondary)
- **Data**: 10 GT-grade sessions from wearable IMU capture (daily motions)
- **Evaluation**: MuJoCo sim2sim, 300-frame windows (10s @ 30fps), fall = root height < 0.45m or roll/pitch > 60°
- **Conditions**: 12 start frames per session (f0 to f600)

### 2.2 Statistical Methods
- Wilson score intervals for fall rates
- McNemar exact test for paired comparisons
- Random forest with GroupKFold (grouped by session) for prediction
- Deduplication: same (session, start, policy) counted once

### 2.3 Feature Definitions (corrected)
From reference CSV columns [Frame, transX, transY, transZ, rotX, rotY, rotZ, joint_1, ..., joint_69]:
- **root_z**: root_translateZ (column 3) — vertical position
- **joint_speed_1s**: mean absolute joint angle change over first 30 frames / 1s
- **root_disp**: Euclidean displacement of (X,Y,Z) over first 30 frames
- **pose_mag**: mean absolute joint angle at first frame
- **pose_var**: variance of joint angles at first frame

## 3. Results

### 3.1 Fall Rate by Start Frame (Policy 035)

| Start | Falls | Rate | Wilson 95% CI | Mean Ori |
|-------|-------|------|---------------|----------|
| f0    | 8/10  | 80%  | [49%, 94%]   | 101°     |
| f50   | 4/10  | 40%  | [17%, 69%]   | 59°      |
| f100  | 5/10  | 50%  | [24%, 76%]   | 74°      |
| f200  | 4/10  | 40%  | [17%, 69%]   | 76°      |
| f400  | 7/10  | 70%  | [40%, 89%]   | 106°     |
| f600  | 9/10  | 90%  | [60%, 98%]   | 144°     |

**Observation**: Rates are higher at f0 and f400-f600, lower at f50-f200. This pattern could reflect initialization sensitivity, content difficulty variation, or both.

### 3.2 Paired McNemar Test (f0 vs f200)

| Session | f0 | f200 | Discordant? |
|---------|-----|------|-------------|
| 135918  | FALL | OK   | Yes (f0 worse) |
| 174102  | FALL | OK   | Yes (f0 worse) |
| 213908  | FALL | OK   | Yes (f0 worse) |
| 214745  | FALL | OK   | Yes (f0 worse) |
| 134446  | FALL | FALL | No |
| 153530  | FALL | FALL | No |
| 154426  | FALL | FALL | No |
| 160809  | FALL | FALL | No |
| 145255  | OK   | OK   | No |
| 151751  | OK   | OK   | No |

Result: b=4 (f0 worse), c=0 (f200 worse), exact p=0.125.

**Interpretation**: All 4 discordant pairs favor mid-motion start. Direction is consistent. However, p=0.125 exceeds conventional α=0.05. With n=10, we cannot reject the null hypothesis of no difference. Approximately 20-25 sessions would be needed for this effect size to reach significance.

### 3.3 Prediction (GroupKFold by Session)

| Metric | Value |
|--------|-------|
| Accuracy (RF, GroupKFold) | 67.5% ± 15.7% |
| Baseline (always predict fall) | 56.7% |
| Improvement over baseline | +10.8 pp |
| n | 120 evaluations |
| Unique sessions | 10 |

**Feature importance** (fit on all data, reference only):
| Feature | Importance |
|---------|-----------|
| pose_variance | 0.270 |
| pose_magnitude | 0.201 |
| root_displacement | 0.151 |
| joint_speed_1s | 0.148 |
| joint_speed_5f | 0.119 |
| root_z | 0.111 |

**Interpretation**: The predictor provides modest improvement over baseline (10.8 pp). The wide CV standard deviation (±15.7%) reflects high variance across session folds. With only 10 sessions, some folds may have as few as 8 training sessions. This predictor should not be used for deployment decisions without further validation.

### 3.4 Cross-Policy Consistency (Exploratory)

Five policies from the same lineage (035 and four fine-tuned variants) show similar patterns of higher fall rates at f0 compared to mid-motion starts. However, these policies share the same base checkpoint and training pipeline; this consistency does not establish universality across independent algorithm families.

## 4. Limitations

1. **Small sample**: n=10 sessions. McNemar test does not reach significance (p=0.125). CI on fall rates are wide (e.g., f0: [49%, 94%]).
2. **Content confound**: Changing start frame changes both initialization AND future content. The observed pattern could reflect either or both.
3. **Single policy lineage**: All tested policies derive from the same 035 checkpoint. Cross-family generalization untested.
4. **Single robot**: Results on A3 only. G1 comparison limited to walking (0 falls, but confounded by different robot/task/policy).
5. **Simulation only**: MuJoCo evaluation; no real-robot validation.
6. **Prediction may not generalize**: 67.5% accuracy with ±15.7% CV variance. Some session folds may drop to near-baseline performance.
7. **Feature interpretation uncertain**: "pose_variance" mixes joint angles of different body parts in different units.

## 5. Proposed Next Steps

### 5.1 Immediate (evidence strengthening)
- Expand to 20-25 sessions for adequate statistical power
- Implement paired design: same future content, different initialization method (isolates content from initialization)
- Report session-level (not evaluation-level) statistics

### 5.2 Mechanism isolation
- Compare: (A) conventional frame-0 reset vs (B) dynamic checkpoint from fixed preroll
- Both track identical future reference frames
- Primary endpoint: fall within fixed horizon
- Analyze: what initial state features predict failure?

### 5.3 Algorithmic contributions (candidate directions)
1. **Confidence-gated onset**: Use predictor to decide whether to apply mitigation
2. **Reference settling**: Prepend brief stabilization period to reference
3. **Multi-frame initialization averaging**: Reduce sensitivity to single-frame noise
4. **Curriculum onset training**: Include mid-motion starts during training

## 6. Conclusion

We present descriptive evidence that humanoid tracking policies may be sensitive to initialization frame choice, with fall rates varying from 40% to 80% depending on start position. The effect direction is consistent but not statistically significant at current sample size. A simple feature-based predictor shows modest accuracy (67.5%). These findings, if confirmed with larger samples and proper mechanism isolation, would have implications for evaluation protocol design and deployment practices.

---

## Data Availability
- Corrected dataset: `reports/onset_corrected.csv` (240 rows, deduplicated)
- Analysis script: `script/onset_analysis_v3.py`
- All metrics: `data/experiments/p2_main/onset_{large,xfine}/`
- Code: github.com/z2964141870-debug/agibot-X2-A3-Q1-model (yuanqi-ops branch)
