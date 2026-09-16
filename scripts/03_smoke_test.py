"""
Day 4 deliverable: smoke_candidates.csv + smoke_metrics.json — a fixed-seed 100-sequence
generation, checked for valid length/alphabet, duplicates, and (once you have a real
training set from Data Engineering) exact training-set matches.
"""

import os
import sys
import json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.model_utils import load_model_and_tokenizer, generate_sequence, MODEL_NAME
from common.validity import is_valid_sequence, summarize_validity

N_SEQUENCES = 100
BASE_SEED = 42
MAX_LEN = 50
TEMPERATURE = 1.0
TOP_P = 0.9


def main():
    model, tokenizer, device = load_model_and_tokenizer()

    records = []
    for i in range(N_SEQUENCES):
        seed = BASE_SEED + i
        seq = generate_sequence(
            model, tokenizer, device,
            max_len=MAX_LEN, temperature=TEMPERATURE, top_p=TOP_P, seed=seed,
        )
        valid, reason = is_valid_sequence(seq)
        records.append({
            "run_id": f"smoke_{i:03d}",
            "sequence": seq,
            "length": len(seq),
            "valid": valid,
            "reason": reason,
            "model_name": MODEL_NAME,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": seed,
        })
        if (i + 1) % 10 == 0:
            print(f"  generated {i + 1}/{N_SEQUENCES}")

    df = pd.DataFrame(records)
    candidates_path = os.path.join(ROOT, "outputs", "smoke_candidates.csv")
    df.to_csv(candidates_path, index=False)

    metrics = summarize_validity(df["sequence"].tolist())
    metrics["model_name"] = MODEL_NAME
    metrics["base_seed"] = BASE_SEED
    metrics["temperature"] = TEMPERATURE
    metrics["top_p"] = TOP_P
    metrics["mean_length"] = round(float(df["length"].mean()), 2)

    metrics_path = os.path.join(ROOT, "outputs", "smoke_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Smoke test summary ---")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {candidates_path}")
    print(f"Wrote {metrics_path}")

    if metrics["validity_rate"] < 0.5:
        print(
            "\nNOTE: validity rate is under 50%. Before assuming the model is bad, check "
            "docs/tokenizer_test.csv — a tokenizer bug is a more common cause than the "
            "model itself at this stage."
        )


if __name__ == "__main__":
    main()
