# Winner decision — primary AR generator

Week 2 deliverable (guide §8): "Write why the winner adds useful candidates and why the
alternative was not selected."

_Fill this in once the baseline and challenger runs are both complete. Do not write it from
intuition — every number below comes from a file in this repo._

## Options compared

| | Baseline | Challenger |
| --- | --- | --- |
| Model | `hugohrban/progen2-small` (151M) | `hugohrban/progen2-medium` (764M) |
| Fine-tuning config | — | — |
| `docs/finetune_manifest*.json` | — | — |

Both were trained on the same splits, the same `data_version`, and generated with the same
sampling code — one variable, so the comparison is fair (guide §5, "fair comparison").

## Evidence

| Metric | Baseline | Challenger | Source |
| --- | --- | --- | --- |
| Final val loss | — | — | `docs/finetune_log*.csv` |
| Validity rate | — | — | `outputs/overlap_*.json` |
| **Novelty rate** | — | — | `docs/novelty_report_*.md` |
| Exact training matches | — | — | `docs/novelty_report_*.md` |
| Near training matches | — | — | `docs/novelty_report_*.md` |
| Effective diversity | — | — | `docs/novelty_report_*.md` |
| Training group coverage | — | — | `docs/novelty_report_*.md` |
| Wall clock (fine-tune) | — | — | `docs/finetune_manifest*.json` |
| Peak GPU memory | — | — | `docs/finetune_manifest*.json` |
| Seconds per sequence | — | — | `docs/sampling_report*.md` |

Remember that validity is near-constant by construction — the sampler makes invalid
sequences unsamplable. Do not use it to separate these two options. Novelty, diversity and
cost are what actually differ.

## Chosen sampling settings

Temperature —, top_p — , from `docs/sampling_report.md`. Selection rule: among grid cells
within 10% of the best validity, take the most diverse.

## Decision

**Primary generator: —**

Why it adds useful candidates:

-

Why the alternative was not selected:

-

Note that "the bigger model scored better on its paper" is not a reason (guide §5). If the
challenger is not clearly better *inside our pipeline*, keep the simpler option.

## Frozen configuration

```bash
# the exact command that produced the final candidate pool
```
