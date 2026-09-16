# Model options — go/no-go

## Baseline: ProGen2-small
- Official repo: https://github.com/salesforce/progen/tree/main/progen2
- Mirror used: hugohrban/progen2-small (HuggingFace) — transformers-native, no JAX needed
- Params: 151M
- License: BSD-3-Clause
- Expected memory: fits comfortably on CPU or a free-tier Colab GPU for smoke-scale generation
- Go/no-go: GO — loaded successfully, see docs/load_test.md

## Challenger (fill in during Week 2)
- Candidate: ProGen2-medium (764M, same mirror family) OR a small causal model trained
  only on the shared peptide view
- Params: —
- License: —
- Expected memory: —
- Go/no-go: — (decide after the baseline comparison in Week 1, per the guide)

## Simple peptide baseline (fallback, Section 5 of the guide)
- A small causal model trained only on the shared AMP view once it exists
- Purpose: transparent speed/data-efficiency baseline, not necessarily the winner
- Status: not started — needs the AR data view from Data Engineering first
