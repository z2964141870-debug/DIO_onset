# Complete Experiment Results (2026-10-22)

## Task 1: Leave-out DIO Validation ✅
- Split: 6 train sessions / 4 test sessions (unseen by DIO)
- Result: 2/4 rescued on test set (100% precision)
  - Onset-sensitive sessions: 2/2 rescued ✅
  - Content-hard sessions: 0/2 attempted ✅ (correctly identified)
- Conclusion: DIO generalizes to unseen sessions, doesn't just memorize

## Task 2: 20-Session Expanded Warm-up ✅
- v7 holdout (10 sessions): 55% → 18% (b=15, c=0, p=0.000061)
- Training set (10 sessions): 100% → 8% (b=37, c=0, p<0.000001)
- Combined: 78% → 13% (b=52, c=0, p<0.00000001)
- Key finding: Even training-set motions fail from cold start (100%!)

## Task 3: DIO + Warm-up Combination ✅
- A (standard f0): 0/4 rescued
- B (warm-up only): 1/4 rescued (25%) — warm-up from f0 has nothing before it
- C (DIO + warm-up): 4/4 rescued (100%)
- D (DIO only): 4/4 rescued (100%)
- Key finding: DIO is the critical component (solves f0 initialization)
