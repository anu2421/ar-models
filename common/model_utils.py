"""
Loading and sampling helpers for the ProGen2-small baseline (hugohrban/progen2-small on
HuggingFace — an unofficial but weight-identical mirror of Salesforce's ProGen2).

ProGen2's tokenizer uses "1" as a start-of-sequence marker meaning "generate N-to-C" and
"2" as the corresponding end token. Confirmed against 02_tokenizer_test.py output.

Sampling controls, in the order they are applied per step:
  1. temperature scaling
  2. allowed-token masking — only the 20 standard amino acids are ever sampleable, plus
     the end token once min_len residues exist. Everything else (B, X, Z, U, O, pad,
     specials) is masked to -inf, so generation cannot emit an automatically-invalid
     residue.
  3. top-p (nucleus) truncation
  4. multinomial draw

KV caching (use_cache=True, feeding only the new token each step) is what makes this
usable at scale.

The standard-amino-acid token id list is resolved ONCE per tokenizer and memoised — it
used to be rebuilt on every single decoding step, which is invisible at 100 sequences and
expensive at 50,000.

NOTE: KV caching assumes this model's forward() accepts `past_key_values` and returns
`outputs.past_key_values`, which is standard for GPT-style HF causal LMs. If a future
transformers version changes the cache object, `03_smoke_test.py` is the canary — a
sudden collapse in sequence diversity means the cache is being ignored or misfed.
"""

import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

from .validity import STANDARD_AMINO_ACIDS, MIN_LENGTH, MAX_LENGTH

MODEL_NAME = "hugohrban/progen2-small"
START_TOKEN = "1"
END_TOKEN = "2"

# Memo table: id(tokenizer) -> sorted list of the 20 standard amino-acid token ids.
_AA_ID_CACHE: dict[int, list[int]] = {}


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    device: str | None = None,
    torch_dtype=None,
):
    """
    torch_dtype: pass torch.bfloat16 to halve the memory used by weights. Note that
    training with bf16 *master* weights is not the same thing as mixed precision — see
    the note in scripts/05_finetune.py. Defaults to fp32 (None), which is what the
    baseline smoke tests use.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch_dtype
    )
    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.no_padding()

    return model, tokenizer, device


def standard_aa_token_ids(tokenizer) -> list[int]:
    """
    Token IDs for exactly the 20 standard amino acids, per this tokenizer's vocab.
    Memoised per tokenizer instance. Raises if any of the 20 is missing — silently
    generating from a 19-letter alphabet would quietly bias every downstream result.
    """
    key = id(tokenizer)
    if key in _AA_ID_CACHE:
        return _AA_ID_CACHE[key]

    vocab = tokenizer.get_vocab()
    missing = sorted(aa for aa in STANDARD_AMINO_ACIDS if aa not in vocab)
    if missing:
        raise ValueError(
            f"These standard amino acids are not single tokens in this tokenizer's "
            f"vocab: {missing}. Generation would silently use a reduced alphabet. "
            f"Inspect the vocab layout before continuing (see 02_tokenizer_test.py)."
        )

    ids = sorted(vocab[aa] for aa in STANDARD_AMINO_ACIDS)
    _AA_ID_CACHE[key] = ids
    return ids


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus filtering. Keeps the smallest set of tokens whose cumulative probability
    reaches top_p, renormalised. Returns (sorted_probs, sorted_idx) so the caller can
    draw and map back.
    """
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # shift by one so the token that crosses the threshold is itself kept
    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return sorted_probs, sorted_idx


def generate_sequence(
    model,
    tokenizer,
    device: str,
    max_len: int = MAX_LENGTH,
    min_len: int = MIN_LENGTH,
    temperature: float = 1.0,
    top_p: float = 0.9,
    seed: int | None = None,
) -> str:
    """
    Autoregressive sampling, one amino acid at a time, with temperature + top-p
    (nucleus) sampling, KV caching, a minimum-length guard, and standard-amino-acid-only
    masking. Stops early once the end token is sampled (only possible after min_len
    residues). Strips start/end markers before returning.
    """
    if seed is not None:
        torch.manual_seed(seed)

    start_id = tokenizer.encode(START_TOKEN).ids
    end_id = tokenizer.encode(END_TOKEN).ids[0]
    standard_ids = standard_aa_token_ids(tokenizer)

    # Precompute the two allowed-id tensors instead of rebuilding a python list per step.
    allowed_no_end = torch.tensor(standard_ids, device=device, dtype=torch.long)
    allowed_with_end = torch.tensor(standard_ids + [end_id], device=device, dtype=torch.long)

    input_ids = torch.tensor([start_id], device=device)
    generated = input_ids
    cur_input = input_ids
    past_key_values = None
    n_residues = 0

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :].float()  # .float() keeps bf16 models stable here
            past_key_values = outputs.past_key_values

            logits = logits / max(temperature, 1e-5)

            allowed = allowed_with_end if n_residues >= min_len else allowed_no_end
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits[:, allowed] = logits[:, allowed]

            probs = torch.softmax(masked_logits, dim=-1)
            sorted_probs, sorted_idx = _top_p_filter(probs, top_p)
            next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_in_sorted)

            generated = torch.cat([generated, next_token], dim=1)
            cur_input = next_token  # KV cache means we only feed the new token from here

            if next_token.item() == end_id:
                break
            n_residues += 1

    token_ids = generated[0].tolist()
    decoded = tokenizer.decode(token_ids)
    return decoded.replace(START_TOKEN, "").replace(END_TOKEN, "").strip()
