# Reproducibility check

Two independent subprocesses, same seed, same settings, compared byte for byte.

## Job

- Model: `outputs/checkpoints/default/epoch_0`
- Sequences: 25
- Base seed: 42 (sequence *i* uses seed 42+*i*)
- temperature 1.3, top_p 0.9
- Device: `cuda`, torch `2.11.0+cu128`
- Python 3.13.15 on Linux-6.6.122+-x86_64-with-glibc2.39

## Result

**IDENTICAL — reproducible**

| Run | SHA256 of concatenated output |
| --- | --- |
| A | `2bd63f6cf1644de924a96a2e71694eeb9bb5c959ce0694771644ed44ac18e073` |
| B | `2bd63f6cf1644de924a96a2e71694eeb9bb5c959ce0694771644ed44ac18e073` |

## Command to reproduce

```bash
python scripts/10_reproducibility_check.py --n 25 --seed 42 --temperature 1.3 --top-p 0.9 \
    --checkpoint outputs/checkpoints/default/epoch_0
```

## What this does and does not prove

It proves the sampling path is deterministic for a fixed seed on THIS device.
It does not prove CPU and GPU agree with each other, nor that results hold
across torch versions. Record the device and version alongside any published
numbers.