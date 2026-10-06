from pathlib import Path
import random

import torch

import data_utils as du
import student.modules as modules
from student.decoder import generate_tokens


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_PATH = Path(__file__).resolve().with_name("model.pt")

def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=device, weights_only=True
    )
    config = checkpoint["model_config"]

    model = modules.TransformerLM(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    data = du.load_memmap_dataset(
        DATA_DIR / "tinystories_valid.bin",
        DATA_DIR / "tinystories_valid.meta",
    )
    vocab, _ = du.load_vocab(DATA_DIR / "tinystories_vocab_merges.pt")
    eos_token_id = next(tok_id for tok_id, tok_bytes in vocab.items()
                        if tok_bytes == b"<|endoftext|>")

    prompt_length = 20
    start = random.randint(0, len(data) - prompt_length)
    prompt_ids = data[start : start + prompt_length].tolist()

    with torch.inference_mode():
        generated_ids = generate_tokens(
            model=model,
            prompt_ids=prompt_ids,
            max_tokens=20,
            temperature=0.8,
            p=0.9,
            eos_token_id=eos_token_id,
            context_length=config['context_length'], 
            device=device,
        )

    print("Prompt:", du.decode(prompt_ids, vocab))
    print("Generated:", du.decode(generated_ids, vocab))


if __name__ == "__main__":
    main()