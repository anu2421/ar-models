# Winner decision — primary AR generator

Week 2 deliverable (guide §8). Every number below comes from a file in this repo.

## Decision

**progen2-small, fine-tuned for 1 epoch (checkpoint `epoch_0`), sampled at temperature 1.3,
top_p 0.9, seed 42.**

Exported as `outputs/ar_candidates.parquet`, run_id `ar_final_ep0_t13`.

## How we got here

The first attempt took the last checkpoint (`epoch_2`) at default temperature 1.0. It
produced a 100% valid pool — and **40% of it was verbatim training sequences**.

The validation curve had already predicted this (`docs/finetune_log.csv`):

| epoch | validation loss |
| --- | --- |
| 0 | 2.5235 |
| 1 | 2.5218 |
| 2 | 2.9910 |

Training loss fell from ~3.07 to ~1.09 over the run while validation loss rose sharply at
epoch 2 — textbook overfitting. The memorisation showed up directly in the novelty
measurement, which is exactly what that measurement exists for.

This is why validity is not a useful objective on its own. The epoch_2 pool scored 100%
validity and 0.96 effective diversity. Both look excellent. Nearly half the pool was copied.

## The sweep

Two independent levers: fewer training epochs, and higher sampling temperature.
Full data in `outputs/sampling_comparison_novelty.csv`.

| checkpoint | T | exact % | near % | **novel %** | eff. diversity | mean length |
| --- | --- | --- | --- | --- | --- | --- |
| epoch_2 | 1.0 | 40 | 54 | 42 | 0.96 | 21.97 |
| epoch_1 | 1.0 | 29 | 44 | 53 | 0.92 | 22.02 |
| epoch_1 | 1.2 | 14 | 24 | 73 | 0.96 | 20.81 |
| epoch_1 | 1.3 | 6 | 16 | 80 | 0.97 | 20.76 |
| epoch_1 | 1.5 | 5 | 8 | 90 | 0.99 | 19.65 |
| epoch_0 | 1.0 | 15 | 18 | 79 | 0.99 | 19.90 |
| **epoch_0** | **1.3** | **4** | **4** | **94** | **0.99** | **20.23** |

Both levers reduce copying monotonically, and they compose: the least-trained checkpoint at
the higher temperature is best on every axis simultaneously.

## Why this configuration

- **Highest novelty measured (94%)** and the lowest exact-copy rate (4%), against 26,699
  training sequences.
- **Mean length 20.23 vs a training mean of 20.85** — the closest match of any high-novelty
  run. Novelty was not bought by drifting off-distribution, which was the failure mode to
  watch for: `epoch_1` at T=1.5 reached 90% novelty but pulled mean length down to 19.65.
- **Effective diversity 0.99** — 99 distinct families in 100 sequences, with zero exact
  duplicates. The pool is not one design with mutations.
- **`epoch_0` costs nothing.** Its validation loss (2.5235) is indistinguishable from
  `epoch_1`'s (2.5218), so preferring the checkpoint that has seen the data less sacrifices
  no measurable fit.

## Why the alternatives were not selected

**Later checkpoints (`epoch_1`, `epoch_2`).** Strictly worse on novelty at every temperature
tested, and `epoch_2`'s validation loss condemns it independently of the novelty
measurement.

**Higher temperature still.** T=1.5 on `epoch_1` gave 90% novelty but moved mean length away
from the training distribution. Beyond roughly T=1.3 the sampler buys novelty by generating
noise, which is not the same as designing something new.

**progen2-medium (challenger).** Not run — compute budget. The baseline reached 94% novelty
at ~0.45 s/sequence on a free-tier T4, and the guide is explicit that a larger model should
not be preferred because its paper reports better numbers. Fine-tuning medium would have
required `--amp` and roughly 4x the wall clock, for a pool whose main weakness — training
overlap — was solved by checkpoint and temperature selection rather than model capacity.
Revisit only if the Shared Evaluator finds this pool weak on predicted activity.

**The `08_sampling_grid.py` recommendation (T=1.0, top_p=0.9).** Not followed. That grid
selects on validity and diversity, and neither discriminates here — validity is ~1.0 in
every cell by construction, and effective diversity is ≥0.94 throughout. The grid could not
see the memorisation, because it does not measure training overlap. That is a real
limitation of the script; `outputs/sampling_comparison_novelty.csv` is the version of this
deliverable that includes novelty, and it is what the decision was made on.

## Runtime and cost

| | value | source |
| --- | --- | --- |
| Fine-tuning wall clock | 27.0 min (3 epochs, 10,014 steps) | `docs/finetune_manifest.json` |
| Peak GPU memory | 3.07 GB | `docs/finetune_manifest.json` |
| Precision | fp32 | `docs/finetune_manifest.json` |
| Generation speed | ~0.45 s/sequence | `outputs/sampling_comparison_finetuned.csv` |
| Hardware | free-tier Colab T4 | — |

Only `epoch_0` is needed for the final pool, so a repeat run costs ~9 minutes rather than 27.

## Reproducibility

`docs/reproducibility_check.md`: two independent subprocesses at the final configuration
produced **byte-identical** output (SHA256 match), on CUDA with torch 2.11.0+cu128.

## Limitations

- **4 of 100 candidates are still verbatim training sequences.** They are exported flagged
  (`exact_training_match`), not dropped — filtering is the Shared Evaluator's decision, and
  the integration guide warns generators against aggressive early filtering.
- **Novelty is edit-distance based, not alignment-based.** `approx_identity_to_nearest` is
  `1 − normalised edit distance` and must not be quoted as an alignment identity.
- **Group-coverage figures in the novelty reports are unreliable for this view.** 85% of the
  training rows (23,458 of 27,509) share a single catch-all `cluster_id`,
  `not_clustered_reference_overlap_or_non_novelty_safe`, so "training groups hit" measures
  almost nothing. Exact/near match rates and effective diversity are unaffected, and are
  what this decision rests on.
- **One seed, 100 sequences per configuration.** The ranking is clear and monotonic in both
  levers, but individual percentages carry a few points of sampling error. A larger pool
  would tighten them.
- **No biophysical or activity evidence.** This decision covers novelty, validity and
  diversity only. Whether these candidates are *good* AMPs is the Shared Evaluator's call.

## Frozen configuration

```bash
# 1. fine-tune (only epoch_0 is needed for the final pool)
python scripts/05_finetune.py \
    --views-dir <DE_REPO>/data/processed/views \
    --epochs 3 --batch-size 8 --lr 5e-5 --seed 42

# 2. generate from the first checkpoint at T=1.3
python scripts/06_generate_from_checkpoint.py \
    --checkpoint outputs/checkpoints/default/epoch_0 \
    --n 100 --temperature 1.3 --top-p 0.9 --seed 42 --tag epoch_0_t13

# 3. measure novelty
python scripts/07_overlap_check.py \
    --candidates outputs/finetuned_candidates_epoch_0_t13.csv \
    --views-dir <DE_REPO>/data/processed/views --tag epoch_0_t13

# 4. export for the Shared Evaluator
python scripts/09_export_candidates.py \
    --candidates outputs/finetuned_candidates_epoch_0_t13.csv \
    --views-dir <DE_REPO>/data/processed/views \
    --checkpoint outputs/checkpoints/default/epoch_0 --run-id ar_final_ep0_t13
```

## Scaling note

This pool is 100 candidates, sized for the comparison. To reach the challenge's target
volume, re-run step 2 with a larger `--n`, generating in batches and checkpointing progress
so an interrupted run can resume (guide §6, Step 6). At ~0.45 s/sequence a 10,000-sequence
pool is roughly 75 minutes on a T4. Re-run steps 3 and 4 on the scaled pool — do not assume
the novelty rate measured at n=100 holds at n=10,000.
