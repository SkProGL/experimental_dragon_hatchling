import os
from pathlib import Path
import json
import time
from contextlib import nullcontext

from counting_models import count_bdh_model
import numpy as np
import torch
from template.gpu_support import GPUSupport

GPUSupport()

# device setup
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

# a2 config
BDH_CONFIG = count_bdh_model.BDHConfig(
    n_layer=8,
    n_embd=384,
    n_head=6,
    mlp_internal_dim_multiplier=64,
    vocab_size=256,
)

BLOCK_SIZE = 128
BATCH_SIZE = 4
MAX_ITERS = 12000
# max_iters = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
EVAL_ITERS = 10

DATA_FILE = "rule_data.txt"
input_file_path = os.path.join(os.path.dirname(__file__), DATA_FILE)

print(f"\033[43m\033[30m{DATA_FILE}\033[0m")

# data


def get_batch(split):
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")

    split_idx = int(0.9 * len(data))
    if split == "train":
        data = data[:split_idx]
    else:
        data = data[split_idx:]

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([
        torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64))
        for i in ix
    ])

    if torch.cuda.is_available():
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# validation
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


def save_model(raw_model):
    CKPT_PATH = Path('counting_models') / "A2_rule_bdh.pt"

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "config": vars(BDH_CONFIG),
    }

    torch.save(checkpoint, CKPT_PATH)
    print(f"Model saved to {CKPT_PATH}")

# time format


def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m >= 60:
        h = m // 60
        m = m % 60
        return f"{h}h {m}m"
    return f"{m}m {s}s"


# save metrics
def save_metrics(data):
    metrics = Path('counting_models') / "bdh_metrics.json"
    with open(metrics, "w") as f:
        json.dump(data, f, indent=4)
    print("Metrics saved to bdh_metrics.json")


# training
def main():
    assert os.path.exists(input_file_path), "rule_data.txt not found"

    start_time = time.time()

    raw_model = count_bdh_model.BDH(BDH_CONFIG).to(device)
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

    loss_acc = 0
    loss_steps = 0

    print("\nStarting training...\n")

    for step in range(MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)

        x, y = get_batch("train")

        loss_acc += loss
        loss_steps += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % LOG_FREQ == 0:
            avg_loss = loss_acc.item() / loss_steps
            val_loss = estimate_val_loss(model)

            print(f"Step {step}/{MAX_ITERS} | train {avg_loss:.4f} | val {val_loss:.4f}")

            steps_log.append(step)
            train_loss_log.append(avg_loss)
            val_loss_log.append(val_loss)

            loss_acc = 0
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
