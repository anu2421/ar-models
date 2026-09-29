# Sampling grid — finetuned

- Model: `outputs/checkpoints/default/epoch_2`
- Device: `cuda`
- 50 sequences per cell, base seed 42 (identical seeds across cells)
- Grid: temperature [0.8, 1.0, 1.2] x top_p [0.85, 0.9, 0.95]

## Results

| T | top_p | validity | unique | eff. diversity | largest family | mean len | s/seq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 0.85 | 1.00 | 1.00 | 0.94 | 4 | 22.64 | 0.503 |
| 0.8 | 0.9 | 1.00 | 1.00 | 0.98 | 2 | 22.92 | 0.502 |
| 0.8 | 0.95 | 1.00 | 1.00 | 0.98 | 2 | 24.2 | 0.526 |
| 1.0 | 0.85 | 1.00 | 1.00 | 0.98 | 2 | 20.2 | 0.441 |
| 1.0 | 0.9 | 1.00 | 1.00 | 1.00 | 1 | 21.96 | 0.48 |
| 1.0 | 0.95 | 1.00 | 1.00 | 1.00 | 1 | 20.34 | 0.461 |
| 1.2 | 0.85 | 1.00 | 1.00 | 1.00 | 1 | 20.66 | 0.472 |
| 1.2 | 0.9 | 1.00 | 1.00 | 1.00 | 1 | 20.28 | 0.464 |
| 1.2 | 0.95 | 1.00 | 1.00 | 1.00 | 1 | 19.96 | 0.447 |

## Recommended setting

**temperature = 1.0, top_p = 0.9**

- validity 1.00, effective diversity 1.00, mean length 21.96

Selection rule: among cells within 10% of the best validity rate, take the most
diverse. Validity on its own is the wrong objective — a cold cell can be perfectly
valid and near-identical, which scores well on validity and badly on novelty.

Because the masking in `common/model_utils.py` makes non-standard residues and
short sequences unsamplable, validity here is expected to sit near 1.0 across the
whole grid. That is the guard working, not the model being good — diversity and the
novelty numbers from `07_overlap_check.py` are what actually discriminate between
these cells.