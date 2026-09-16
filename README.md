# AR Models Role — AMP Challenge 2026

Your Week 1 deliverables (per the Autoregressive Models role guide), turned into a runnable
mini-project. Everything here does the Day 1–4 tasks: pick a baseline, load it, test the
tokenizer, and run a 100-sequence smoke test — *before* the shared dataset view exists.

## 0. Baseline choice

Baseline: `hugohrban/progen2-small` — a HuggingFace mirror of Salesforce's ProGen2-small
(151M params), the safest first feasibility test the guide recommends. It's easy to load with
`transformers`, unlike the official repo which needs JAX. This satisfies the guide's Day 1
requirement to verify "official code, weights, license, model size and expected memory" —
just make sure you record in `docs/model_options.md` that you're using this mirror and why
(no JAX dependency, same weights, actively maintained).

Challenger (Week 2, not built yet): either `progen2-medium` (same mirror family, 764M params)
or a small causal model trained only on the shared peptide view once Data Engineering hands
that over. Don't build the challenger yet — Week 1 is baseline-only per the guide.

## 1. Set up the environment

```bash
cd ar-models
conda env create -f environment.yml
conda activate ar-models
```

No conda? Use venv instead:
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Known issue:** `transformers` 5.0+ removed some internal methods (`get_head_mask` among
them) that ProGen2's custom model code still uses via `trust_remote_code=True`. If
`01_load_test.py` fails with `AttributeError: ... has no attribute 'get_head_mask'`, run
`pip install "transformers<5.0"` and try again. `requirements.txt`/`environment.yml`
already pin this for fresh installs.

GPU is optional for a 151M model — it'll run on CPU for smoke-testing, just slower
(expect a few seconds per sequence on CPU vs milliseconds on GPU). If your machine has no
GPU, run this from Google Colab instead (free T4 GPU) — just `!pip install -r requirements.txt`
in the first cell and copy the `common/` and `scripts/` folders in.

## 2. Run order (matches the guide's Days 1–4)

```bash
python scripts/01_load_test.py        # Day 1-2: does the model load at all?
python scripts/02_tokenizer_test.py   # Day 3: do the 20 amino acids survive encode/decode?
python scripts/03_smoke_test.py       # Day 4: generate 100 sequences, fixed seed, measure validity
```

Each script writes its deliverable into `docs/` or `outputs/`, matching the file names the
guide asks for.

## 3. What's NOT here yet (comes later)

- `data/` is empty on purpose. Week 1's `ar_data_requirements.md` (Day 5 deliverable) should
  list what you need from Data Engineering — see the template in `docs/`. Don't build your own
  cleaned peptide file; that duplicates their work and the guide explicitly says not to.
- The challenger model (Day 6) and evaluator connection (Day 7) aren't scaffolded — build
  those once `01`–`03` run cleanly and you've picked your challenger.
- Fine-tuning code isn't here yet either — Week 1 is baseline-only. Add a `train/` folder in
  Week 2 once you've confirmed fine-tuning is worth the compute (see the guide's Section 5).

## 4. VS Code setup

1. Open the `ar-models/` folder directly (File → Open Folder), not a parent folder — keeps
   relative paths working.
2. Install the **Python** and **Jupyter** extensions if you don't have them.
3. `Ctrl+Shift+P` → "Python: Select Interpreter" → pick the `ar-models` conda env or `.venv`.
4. Run scripts with the ▷ button, or `Ctrl+F5`, or just `python scripts/01_load_test.py` in
   the integrated terminal (\`Ctrl+\`\`).
5. First run downloads the model (~600MB) from HuggingFace — it caches in `~/.cache/huggingface`,
   so it's a one-time wait.
