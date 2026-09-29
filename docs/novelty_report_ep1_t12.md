# Novelty report — ep1_t12

- Candidate file: `outputs/finetuned_candidates_ep1_t12.csv`
- Compared against: splits ['core_train_only', 'train'] of the AR view (`26699` sequences)
- data_version: `1.0`
- Near-match threshold: edit distance <= 1; identity flag at >= 0.9

## Headline numbers

| Metric | Count | Rate |
| --- | --- | --- |
| Candidates | 100 | — |
| Valid (challenge rules) | 100 | 100.0% |
| Exact training matches | 29 | 29.0% |
| Near training matches (<= 1 edit) | 44 | 44.0% |
| High identity (>= 0.9) | 47 | 47.0% |
| **Novel** | **53** | **53.0%** |

## Family collapse within the pool

- Exact duplicates: 2
- Clusters at edit distance <= 2: 92 for 100 candidates
- Effective diversity (clusters / candidates): **0.92** (1.0 = every candidate its own family)
- Largest single family: 4 candidates
- Singleton rate: 88.0%

Largest families (representative sequence):

- `ILPWKGPWWPWRR` — 4 members
- `GFCWNVCVYRNGVRVCHRRCN` — 4 members
- `NLCEKASKTWSGNCGNTGHCDNQCRSWEGAKHGACHVRNGKHKCFCYFNC` — 2 members
- `VRLRIRVAVILA` — 2 members

## Coverage of training similarity groups

- Training groups hit: 8 / 743 (1.1%)
- Candidates not attributable to any training group: 50 (50.0%) — a high number here is a GOOD novelty signal

## How to read this

A fluent copy is not a new design. Exact and near matches should both be low; if the
near-match rate is high while validity looks great, the model has memorised the
training corpus rather than learned the design space. Effective diversity well below
1.0 means the pool is a handful of families with point mutations — fix that by raising
sampling temperature or reducing fine-tuning epochs, not by generating more sequences.

Caveat: identity here is `1 - normalised edit distance`, not a Smith-Waterman
alignment identity. Do not quote these numbers as alignment identities in the
final write-up.