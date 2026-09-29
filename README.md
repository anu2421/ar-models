# AR Models Role — AMP Challenge 2026

Autoregressive generation of antimicrobial peptide candidates, for the AMP Challenge 2026
team project. This repo covers the Autoregressive Models role: pick a baseline, fine-tune
it on the shared peptide view, measure whether its output is actually *new*, and hand a
schema-conformant candidate pool to the Shared Evaluator role.

**Owner:** @anu2421 · **Domain tag:** `ar` · **Deadline:** 1 October 2026 (AoE)

---

## 1. What this repo produces

| Guide deliverable | Produced by | Lands in |
| --- | --- | --- |
| `model_options.md` (go/no-go) | written by hand | `docs/model_options.md` |
| `load_test.md` | `01_load_test.py` | `docs/load_test.md` |
| `tokenizer_test.csv` | `02_tokenizer_test.py` | `docs/tokenizer_test.csv` |
| `length_test.md` | `02_tokenizer_test.py` | `docs/length_test.md` |
| `smoke_candidates.csv` + `smoke_metrics.json` | `03_smoke_test.py` | `outputs/` |
| `ar_data_requirements.md` | written by hand | `docs/ar_data_requirements.md` |
| AR view inspection | `04_inspect_ar_view.py` | `docs/ar_view_inspection.md` |
| run manifest + training log | `05_finetune.py` | `docs/finetune_manifest.json`, `docs/finetune_log.csv` |
| fine-tuned candidate pool | `06_generate_from_checkpoint.py` | `outputs/finetuned_candidates.csv` |
| **novelty / training-overlap report** | `07_overlap_check.py` | `docs/novelty_report_*.md`, `outputs/overlap_*.csv` |
| `sampling_comparison.csv` | `08_sampling_grid.py` | `outputs/sampling_comparison.csv`, `docs/sampling_report.md` |
| `ar_candidates.parquet` + `generation_manifest.json` | `09_export_candidates.py` | `outputs/` |
| `reproducibility_check.md` | `10_reproducibility_check.py` | `docs/` |
| `winner_decision.md` | written by hand, from the reports above | `docs/winner_decision.md` |

---

## 2. Model choice

**Baseline: `hugohrban/progen2-small`** — a HuggingFace mirror of Salesforce's ProGen2-small
(151M params). The official Salesforce repo needs JAX; this mirror is `transformers`-native
with the same weights, which is the only reason it is preferred. License: BSD-3-Clause
(Salesforce's original). Recorded in `docs/model_options.md` as the guide's Day 1 requires.

**Challenger: `hugohrban/progen2-medium`** (764M). Run every script against it with
`--model-name hugohrban/progen2-medium --tag medium`; outputs go to separate filenames so a
challenger run never overwrites baseline results.

---

## 3. Setup

```bash
git clone https://github.com/anu2421/ar-models.git
cd ar-models
conda env create -f environment.yml
conda activate ar-models
```

No conda:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Known issue — `transformers` 5.0+:** ProGen2's `trust_remote_code=True` model code calls
internal methods (`get_head_mask` among them) that were removed in transformers 5. If
`01_load_test.py` raises `AttributeError: ... has no attribute 'get_head_mask'`, run
`pip install "transformers<5.0"`. `requirements.txt` and `environment.yml` already pin this.

**Hardware.** Generation with progen2-small runs fine on CPU (seconds per sequence).
Fine-tuning wants a GPU. On a free-tier Colab T4, progen2-small fits in fp32; progen2-medium
needs `--amp` (see §6). First run downloads ~600MB from HuggingFace into
`~/.cache/huggingface`.

---

## 4. Getting the data

This repo contains **no peptide data** and never should — Data Engineering owns the single
cleaned dataset, and duplicating it breaks the traceability the challenge requires. You need
their `data/processed/views/` directory, and every script that touches data takes
`--views-dir` pointing at it.

Columns this repo expects (requested in `docs/ar_data_requirements.md`):
`sequence`, `length`, `split`, `similarity_group_id`, `data_version`.

Common alternate names (`cluster_id`, `seq`, `seq_len`, `version`, …) are renamed
automatically on load — see `COLUMN_ALIASES` in `common/data_view.py`. Check what actually
arrived before training:

```bash
python scripts/04_inspect_ar_view.py --views-dir /path/to/amp-data/data/processed/views
```

If auto-detection picks the wrong file, pass the exact path with `--view-path`. That is
always safer than pattern matching.

---

## 5. Run order

### Week 1 — feasibility, no data needed

```bash
python scripts/01_load_test.py          # does the model load on this hardware?
python scripts/02_tokenizer_test.py     # do all 20 amino acids survive encode/decode?
python scripts/03_smoke_test.py         # 100 sequences, fixed seed, validity measured
```

### Week 2 — the real corpus

```bash
VIEWS=/path/to/amp-data/data/processed/views

python scripts/04_inspect_ar_view.py --views-dir $VIEWS

# quick 30-step sanity run first — it still writes a checkpoint and a manifest
python scripts/05_finetune.py --views-dir $VIEWS --max-steps 30

# the real run
python scripts/05_finetune.py --views-dir $VIEWS --epochs 3 --batch-size 8 --lr 5e-5

python scripts/06_generate_from_checkpoint.py \
    --checkpoint outputs/checkpoints/default/epoch_2 --n 100 --tag finetuned

# pick sampling settings with evidence, not by guessing
python scripts/08_sampling_grid.py \
    --checkpoint outputs/checkpoints/default/epoch_2 --n 50 --tag finetuned
```

### Before any export — the check that decides whether the work counts

```bash
python scripts/07_overlap_check.py \
    --candidates outputs/finetuned_candidates.csv \
    --views-dir $VIEWS --tag finetuned
```

### Handover to the Shared Evaluator

```bash
python scripts/09_export_candidates.py \
    --candidates outputs/finetuned_candidates.csv \
    --views-dir $VIEWS \
    --checkpoint outputs/checkpoints/default/epoch_2 \
    --run-id ft_run1

python scripts/10_reproducibility_check.py \
    --checkpoint outputs/checkpoints/default/epoch_2 --n 25
```

---

## 6. Decisions worth knowing before you change anything

**Training split = `core_train_only` + `train` (26,699 sequences).** Team decision, Sept
2026: `train` alone was judged too small for a 151M-param model. `validation` and `test`
stay held out. Change `TRAIN_SPLITS` in `05_finetune.py` if the team revisits this — and
say so in the manifest.

**Sampling is guarded, so validity is not a meaningful score.** `generate_sequence` masks
every non-standard token out of the distribution and masks the end token until `min_len`
residues exist. A sequence that violates the alphabet or length rules is therefore
*unsamplable*, and validity sits near 100% by construction. Anyone reading a 99% validity
rate as evidence the model is good has misread it. **Novelty and diversity are the numbers
that discriminate** — see `07_overlap_check.py`.

**Precision.** Plain fp16 master weights produced NaN loss during testing (Sept 2026). The
first fix was to load master weights in bf16, which stops the NaN but trains AdamW on bf16
weights — at lr 5e-5, updates approach the edge of bf16's 8-bit mantissa and small updates
round away. `--amp` is the correct fix (fp32 master weights, bf16 autocast, GradScaler) and
is what you should use for progen2-medium. `--bf16-weights` is kept only to reproduce
earlier runs.

**Over-length sequences are dropped, not truncated.** Truncating at `MAX_SEQ_LEN` cuts the
END token off the row, which teaches the model never to stop. The challenge caps valid
peptides at 50 residues, so a longer row is out-of-spec input rather than data to salvage.
`05_finetune.py` reports how many rows it dropped.

**We do not rank our own candidates.** `raw_score` is exported empty and clearly labelled.
Scoring and selection belong to the Shared Evaluator role; a generator applying a private
ranking is exactly what the integration guide warns against. Invalid sequences are exported
too, flagged rather than dropped, so the evaluator sees the true pool.

---

## 7. Reading the novelty report

`07_overlap_check.py` is the deliverable that answers the challenge's completion criterion
*"training matches, duplicates and family collapse were measured."* It measures three
different things:

- **Exact match** — the candidate is verbatim in the training corpus.
- **Near match** — within `--max-edits` (default 1) of a training sequence. A single point
  mutation off a known AMP is not a new design.
- **Family collapse** — the pool's own internal redundancy. 1,000 sequences that are really
  40 designs with point mutations will show a high `largest_family` and an
  `effective_diversity` well below 1.0. A plain duplicate count will not catch this.

Rough thresholds to worry at: exact match rate above 1%, near match rate above 20%,
effective diversity below 0.5. The script prints warnings at those levels.

Identity is reported as `1 − normalised edit distance`, **not** a Smith-Waterman alignment
identity. Do not quote these as alignment identities in the final write-up.

---

## 8. Results

_Fill this in after the real runs — a reviewer cloning this repo should see the numbers here
without running anything._

| Run | Model | Validity | Novelty | Effective diversity | Mean length |
| --- | --- | --- | --- | --- | --- |
| Baseline smoke | progen2-small (pretrained) | — | — | — | — |
| Fine-tuned | progen2-small, 3 epochs | — | — | — | — |
| Challenger | progen2-medium | — | — | — | — |

Sampling settings chosen: _T = —, top_p = —_ (justified in `docs/sampling_report.md`).
Winner and reasoning: `docs/winner_decision.md`.

---

## 9. Reproducing a run exactly

```bash
python scripts/10_reproducibility_check.py --n 25 --seed 42
```

Runs the same small job in two separate subprocesses and compares SHA256 hashes. Two
processes rather than two loops, because that is what catches hidden global state. Exits
non-zero on divergence and reports the first differing index.

Every generated sequence records its own seed; sequence *i* uses `base_seed + i`. Determinism
holds per device — CPU and GPU draw differently, so record which one produced any published
number. `09_export_candidates.py` writes `outputs/SHA256SUMS` for the export itself.

---

## 10. Repo layout

```
common/
  validity.py      shared challenge rules (20 AAs, 8-50 residues) — keep identical to Data Engineering's
  model_utils.py   model loading + guarded autoregressive sampling
  data_view.py     locating, loading and normalising the AR view
  novelty.py       edit distance, training overlap, family-collapse clustering
scripts/           01-10, run in order (see §5)
docs/              deliverables that are documents
outputs/           deliverables that are data (gitignored except the small committed ones)
data/              stays empty — Data Engineering owns the data
```

---

## 11. Limitations

- Novelty uses edit distance, not sequence alignment. For peptides of similar length it
  tracks alignment identity closely enough to flag near-copies, but it is not a substitute
  for a proper alignment-based novelty check.
- Validity here is the challenge's hard rules only (alphabet + length). Biophysical
  plausibility, activity and toxicity are the Shared Evaluator's job, and nothing in this
  repo predicts them.
- The `hugohrban` mirror is unofficial. Weights are believed identical to Salesforce's
  release; this has not been independently verified checksum-for-checksum.
- No structural validation of any kind.
- Determinism is verified per device, not across devices or torch versions.

---

## 12. VS Code

1. Open the `ar-models/` folder directly (File → Open Folder), not a parent — keeps relative
   paths working.
2. Install the **Python** extension.
3. `Ctrl+Shift+P` → "Python: Select Interpreter" → the `ar-models` conda env or `.venv`.
4. Run with the ▷ button, or `python scripts/01_load_test.py` in the integrated terminal.

## License

MIT — see `LICENSE`. Model weights are covered by their own license (ProGen2: BSD-3-Clause).
