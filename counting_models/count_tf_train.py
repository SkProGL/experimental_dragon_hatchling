import os
import json
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from template.gpu_support import GPUSupport

GPUSupport()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)

scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device} with dtype {dtype}")

# config (match bdh a2)
BLOCK_SIZE = 128
BATCH_SIZE = 4
MAX_ITERS = 12000
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
EVAL_ITERS = 10

N_LAYER = 8
N_HEAD = 6
N_EMBD = 384
VOCAB_SIZE = 256
DROPOUT = 0.1

DATA_FILE = "rule_data.txt"
input_file_path = os.path.join(os.path.dirname(__file__), DATA_FILE)

print(f"\033[43m\033[30m{DATA_FILE}\033[0m")


# model
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_embed = nn.Parameter(torch.zeros(1, BLOCK_SIZE, N_EMBD))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=N_EMBD,
            nhead=N_HEAD,
            dim_feedforward=N_EMBD * 4,
            dropout=DROPOUT,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYER
        )

        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        _, T = idx.shape

        if T > BLOCK_SIZE:
            idx = idx[:, -BLOCK_SIZE:]
            T = BLOCK_SIZE

        x = self.embed(idx) + self.pos_embed[:, :T, :]

        # causal mask: true means blocked
        causal_mask = torch.triu(
            torch.ones(T, T, device=idx.device, dtype=torch.bool),
            diagonal=1
        )

        x = self.transformer(x, mask=causal_mask)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if targets.shape[1] > T:
                targets = targets[:, -T:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=1):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx


# data
def get_batch(split):
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")

    split_idx = int(0.9 * len(data))
    if split == "train":
        data = data[:split_idx]
    else:
        data = data[split_idx:]

    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))

    x = torch.stack([
        torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64))
        for i in ix
    ])

    if torch.cuda.is_available():
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_val_loss(model):
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch("val")
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m >= 60:
        h = m // 60
        m = m % 60
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def save_metrics(data):
    path = Path("counting_models") / "transformer_metrics.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Metrics saved to {path}")


def save_model(raw_model):
    path = Path("counting_models") / "A2_rule_transformer.pt"
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "config": {
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "max_iters": MAX_ITERS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "log_freq": LOG_FREQ,
            "eval_iters": EVAL_ITERS,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_embd": N_EMBD,
            "vocab_size": VOCAB_SIZE,
            "dropout": DROPOUT,
            "data_file": DATA_FILE,
        },
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


# train
def main():
    assert os.path.exists(input_file_path), f"{DATA_FILE} not found"

    start_time = time.time()

    raw_model = TransformerModel().to(device)
    model = torch.compile(raw_model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    steps_log = []
    train_loss_log = []
    val_loss_log = []

    x, y = get_batch("train")

    loss_acc = 0.0
    loss_steps = 0

    print("\nStarting training...\n")

    for step in range(MAX_ITERS):
        with ctx:
            _, loss = model(x, y)

        x, y = get_batch("train")

        loss_acc += loss.item()
        loss_steps += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % LOG_FREQ == 0:
            avg_train_loss = loss_acc / loss_steps
            val_loss = estimate_val_loss(model)

            print(f"[TF] Step {step}/{MAX_ITERS} | train {avg_train_loss:.4f} | val {val_loss:.4f}")

            steps_log.append(step)
            train_loss_log.append(avg_train_loss)
            val_loss_log.append(val_loss)

            loss_acc = 0.0
            loss_steps = 0

    elapsed = time.time() - start_time

    save_metrics({
        "steps": steps_log,
        "train_loss": train_loss_log,
        "val_loss": val_loss_log,
        "time_seconds": elapsed,
        "time_formatted": format_time(elapsed),
    })

    save_model(raw_model)


if __name__ == "__main__":
    main()
