# Model options — go/no-go

Day 1 deliverable (guide §7). Records the official code, weights, license, model size and
expected memory for every option considered, plus a go/no-go note.

## Baseline: ProGen2-small — **GO**

| Field | Value |
| --- | --- |
| Official repo | https://github.com/salesforce/progen/tree/main/progen2 |
| Weights used | `hugohrban/progen2-small` (HuggingFace mirror) |
| Why the mirror | `transformers`-native; the official repo requires JAX. Same weights. |
| Parameters | 151M |
| License | BSD-3-Clause (Salesforce) |
| Memory, generation | fits on CPU; ~1GB GPU |
| Memory, fine-tuning | fp32 fits a free-tier T4 at batch size 8 |
| Go/no-go | **GO** — loads and generates, see `docs/load_test.md` |

Caveat: the mirror is unofficial. Weights are believed identical to the Salesforce release
but this has not been verified checksum-for-checksum.

## Challenger: ProGen2-medium — **GO**

| Field | Value |
| --- | --- |
| Weights used | `hugohrban/progen2-medium` |
| Parameters | 764M |
| License | BSD-3-Clause (Salesforce) |
| Memory, generation | ~3GB GPU |
| Memory, fine-tuning | needs `--amp`; fp32 will OOM on a T4 |
| Go/no-go | **NOT RUN** — compute budget, see below |

**Outcome: not run.** The baseline reached 94% novelty at ~0.45 s/sequence on a free-tier
T4 (`docs/winner_decision.md`). Fine-tuning medium needs `--amp` and roughly 4x the wall
clock, and the pool's main weakness — training overlap — was solved by checkpoint and
temperature selection rather than model capacity. The guide (§5) is explicit that a larger
model should not be preferred because its paper reports better numbers, and that the simpler
option should be kept when the complex one does not add reliable value inside our own
pipeline. Revisit only if the Shared Evaluator finds this pool weak on predicted activity.

## Considered and not selected

**A small causal model trained from scratch on the shared AMP view only.** The guide's §5
fallback — transparent and data-efficient, but with ~27k training sequences it cannot match
a model pretrained on millions of proteins, and the compute is better spent on the
fine-tuning comparison. Revisit only if both ProGen2 options turn out to memorise the
training corpus (watch the near-match rate in `docs/novelty_report_*.md`).

## Decision

Baseline and challenger are the same architecture at two sizes, which makes the comparison
clean: same tokenizer, same sampling code, same data, one variable. The final choice goes in
`docs/winner_decision.md` with the evidence from the sampling grid and the novelty reports.
