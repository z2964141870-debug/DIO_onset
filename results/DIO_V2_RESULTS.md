
# DIO v2 Evaluation Results (2026-10-22)

## Model
- Architecture: Transformer(256) + MLP(512) denoiser, 100 diffusion steps
- Training: 500 augmented samples, 2000 epochs, 257s on RTX 3090
- Final loss: 0.0586 (vs 0.88 in v1)

## Results: 4/8 rescued (100% of onset-sensitive sessions)

| Session | Original | DIO v2 | Classification |
|---------|----------|--------|----------------|
| 135918 | 110° FALL | 8° SURVIVE | onset-sensitive ✅ |
| 174102 | 134° FALL | 18° SURVIVE | onset-sensitive ✅ |
| 213908 | 147° FALL | 16° SURVIVE | onset-sensitive ✅ |
| 214745 | 131° FALL | 21° SURVIVE | onset-sensitive ✅ |
| 134446 | 107° FALL | 112° FALL | content-hard ❌ |
| 153530 | 124° FALL | 94° FALL | content-hard ❌ |
| 154426 | 133° FALL | 124° FALL | content-hard ❌ |
| 160809 | 103° FALL | 78° FALL | content-hard ❌ |

## Key Findings
1. DIO v2 rescues ALL onset-sensitive sessions (4/4)
2. DIO v2 cannot rescue content-hard sessions (0/4, expected)
3. Orientation improvement: 84-93% reduction in rescued sessions
4. DIO v2 >> DIO v1 (50% vs 0% rescue rate)

