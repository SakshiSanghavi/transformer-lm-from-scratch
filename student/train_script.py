from pathlib import Path

import torch

import data_utils as du
import student.modules as modules
import student.training as training
import matplotlib.pyplot as plt
from student.decoder import generate_tokens


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_PATH = Path(__file__).resolve().with_name("model.pt")

MODEL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 128,
    "num_layers": 4,
    "d_model": 128,
    "num_heads": 8,
    "d_ff": 384,
    "theta": 10_000.0,
}


def main():
    device = ("cuda" if torch.cuda.is_available() 
              else "mps" if torch.backends.mps.is_available() 
              else "cpu")
    print(f"Using device: {device}")

    data = du.load_memmap_dataset(
        DATA_DIR / "tinystories_valid.bin",
        DATA_DIR / "tinystories_valid.meta",
    )

    model = modules.TransformerLM(**MODEL_CONFIG).to(device)
    model.to(device=device)

    vocab, _ = du.load_vocab(DATA_DIR / "tinystories_vocab_merges.pt")
    eos_token_id = next(
        token_id for token_id, token_bytes in vocab.items()
        if token_bytes == b"<|endoftext|>"
    )

    prompt_ids = data[:20].tolist()

    def show_sample(label):
        model.eval()
        with torch.inference_mode():
            generated_ids = generate_tokens(
                model=model,
                prompt_ids=prompt_ids,
                max_tokens=40,
                temperature=0.8,
                p=0.9,
                eos_token_id=eos_token_id,
                context_length=MODEL_CONFIG["context_length"],
                device=device,
            )
        print(f"\n{label}")
        print("Prompt:", du.decode(prompt_ids, vocab))
        print("Generated:", du.decode(generated_ids, vocab))

    show_sample("Before training")
    model.train()
    optimizer = training.AdamW(model.parameters(),lr=1e-3)
    batch_size = 32
    steps=600 
    losses = []
    for batch_id in range(steps):
        inputs, targets = training.get_batch(data,batch_size,MODEL_CONFIG['context_length'],device)
        optimizer.zero_grad()
        loss = training.cross_entropy(model(inputs),targets)
        loss.backward()
        training.gradient_clipping(model.parameters(),max_l2_norm=1.0)
        optimizer.step()
        # print(loss.item())
        losses.append(loss.item())
        if batch_id % 50 == 0:
            print(f"Batch {batch_id}/{steps}, loss: {loss.item():.4f}")

    torch.save(
        {
            "model_config": MODEL_CONFIG,
            "model_state_dict": model.state_dict(),
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")

    step_numbers = range(1, len(losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(step_numbers, losses, label="Raw Loss", alpha=0.35, color="dodgerblue")

    window = 20
    if len(losses) >= window:
        smoothed_losses = [
            sum(losses[i : i + window]) / window
            for i in range(len(losses) - window + 1)
        ]
        smoothed_steps = range(window, len(losses) + 1)
        plt.plot(
            smoothed_steps,
            smoothed_losses,
            label=f"Smoothed ({window}-step avg)",
            color="firebrick",
            linewidth=2,
        )

    plt.title("Transformer Training Loss")
    batch_numbers = range(1, len(losses) + 1)
    plt.plot(batch_numbers, losses, label="Loss per batch", alpha=0.35, color="dodgerblue")
    plt.xlabel("Batches processed")    
    plt.ylabel("Loss")
    plt.savefig("learning_curve.png", dpi=300)

    print("Done")

    show_sample("After training")


if __name__ == "__main__":
    main()