# DIO: Diffusion Initialization Optimizer for Humanoid Motion Tracking

## What is this?

We discovered that humanoid motion tracking policies are extremely sensitive to their initialization state. When a policy starts tracking from "cold" (no motion context), it falls **60-76%** of the time. With a simple 50-frame warm-up, this drops to **22-32%** (McNemar test, p < 0.0001, n=50 pairs).

## Key Findings

### 1. Cold-Start is the Root Cause
- 65% of all falls occur within the first 2 seconds of tracking
- The policy's recurrent state (RNN hidden) starts with zero motion context
- This is NOT a content difficulty problem — it's a context problem

### 2. Warm-up Effectively Resolves It
| Metric | Cold Start | With Warm-up | Reduction |
|--------|-----------|-------------|-----------|
| Policy 035 fall rate | 60% | 22% | ↓38pp |
| Policy M17 fall rate | 76% | 32% | ↓44pp |
| McNemar p-value | — | < 0.0001 | Significant |
| Reverse pairs (got worse) | — | 0/50 | No side effects |

### 3. Session Taxonomy
We classify tracking failures into two types:
- **Onset-sensitive** (cold-start): Rescued by warm-up (19-22/50)
- **Content-hard**: Fall regardless of warm-up (11-16/50)

## DIO: Diffusion Initialization Optimizer

We propose using a conditional diffusion model to generate optimal initialization states:
- **Input**: First 30 frames of reference motion (joints + velocities)
- **Output**: Optimal initial state (31 joint angles + 31 velocities)
- **Architecture**: Transformer encoder + MLP denoiser (DDPM)
- **Training**: Denoising score matching on successful episodes

## Repository Structure

```
src/
  dio_model.py          # DIO diffusion model (v1 architecture)
  dio_v2_train.py       # Improved training (data augmentation, 2000 epochs)
  dio_v2_eval.py        # Evaluation with warm-up trajectory generation
  warmup_exp.py         # Warm-up experiment (pilot, n=10)
  warmup_large.py       # Large-scale warm-up (n=50 pairs × 2 policies)
  onset_analysis_v3.py  # Corrected analysis (Wilson CI, GroupKFold)
  extract_dio_data.py   # Training data extraction

experiments/
  (evaluation configs and reference data paths)

docs/
  ONSET_RESEARCH_SUMMARY_V2.md    # Rigorous summary (corrected numbers)
  LITERATURE_SURVEY_UPDATED.md    # 8-paper comparison (SuperMimic, RGMT, etc.)
  DEEP_INNOVATION.md              # Innovation directions analysis
  MODERN_METHODS_INNOVATION.md    # Diffusion/Transformer/Flow methods
  EXECUTION_PLAN.md               # 4-week execution plan
  EXPERIMENT_DESIGN_ISOLATION.md  # 2×2 paired experiment design
  DIO_NEGATIVE_RESULT.md          # V1 negative result analysis

results/
  onset_corrected.csv             # 240 deduplicated evaluations
  warmup_large.log                # Full warm-up experiment output
  ONSET_FINAL_RESULTS.txt         # Final analysis numbers
```

## Quick Start

```bash
# 1. Extract training data from evaluations
python src/extract_dio_data.py

# 2. Train DIO (v2, with augmentation)
python src/dio_v2_train.py

# 3. Evaluate DIO with warm-up trajectories
python src/dio_v2_eval.py

# 4. Run large-scale warm-up validation
python src/warmup_large.py
```

## Related Work

| Paper | Method | Relation to ours |
|-------|--------|-----------------|
| RGMT (arXiv 2601.23080) | Random unstable init (training) | They train with random init; we analyze deployment-time warm-up |
| SuperMimic (CoRL 2026 Oral) | Differentiable world model | They solve broader failure; we analyze onset specifically |
| BeyondMimic | Diffusion-guided control | Uses phase sampling; we quantify initialization sensitivity |
| SONIC/PHC | Model-free RL tracking | Standard frame-0 init; we show this causes systematic bias |

## Citation

```bibtex
@article{dio2026,
  title={Diffusion Initialization Optimization for Humanoid Motion Tracking},
  author={Yu, Qi and collaborators},
  year={2026},
  note={arXiv preprint}
}
```

## License

TBD
