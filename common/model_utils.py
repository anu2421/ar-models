"""
Loading and sampling helpers for the ProGen2-small baseline (hugohrban/progen2-small on
HuggingFace — an unofficial but weight-identical mirror of Salesforce's ProGen2).

IMPORTANT: ProGen2's tokenizer uses "1" as a start-of-sequence marker meaning "generate
N-to-C" and "2" as the corresponding end token (there's also a reverse "2 ... 1" direction
for C-to-N, which we don't use here). Confirm this against your own
`02_tokenizer_test.py` output before trusting it in a real run — vocab details can differ
between mirrors and model versions, which is exactly why the guide has you write a
tokenizer test before generating anything.
"""

import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

MODEL_NAME = "hugohrban/progen2-small"
START_TOKEN = "1"
END_TOKEN = "2"


def load_model_and_tokenizer(model_name: str = MODEL_NAME, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.no_padding()

    return model, tokenizer, device


def generate_sequence(
    model,
    tokenizer,
    device: str,
    max_len: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    seed: int | None = None,
) -> str:
    """
    Autoregressive sampling, one amino acid at a time, with temperature + top-p (nucleus)
    sampling. Stops early if the model emits the end token. Strips start/end markers
    before returning, so the output is a plain amino-acid string ready for validity
    checking.
    """
    if seed is not None:
        torch.manual_seed(seed)

    start_id = tokenizer.encode(START_TOKEN).ids
    end_id = tokenizer.encode(END_TOKEN).ids[0]
    input_ids = torch.tensor([start_id], device=device)

    generated = input_ids
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(generated).logits[:, -1, :]
            logits = logits / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # keep the smallest set of tokens whose cumulative prob exceeds top_p
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_in_sorted)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == end_id:
                break

    token_ids = generated[0].tolist()
    decoded = tokenizer.decode(token_ids)
    # strip any leftover start/end marker characters
    return decoded.replace(START_TOKEN, "").replace(END_TOKEN, "").strip()
