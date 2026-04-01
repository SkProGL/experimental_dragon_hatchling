# Copyright Pathway Technology, Inc.
from pathlib import Path
from template.gpu_support import GPUSupport
from factual_models import tuning_model
from factual_models import load_dataset
from factual_models.tf_model import Transformer

import math
import copy
import numpy as np
from contextlib import nullcontext
import os
import torch
import pandas as pd


# print(os.listdir())
# with open('run.txt', 'r')as f:
#     run = f.read()
#     print(f"{run}")
run_config = tuning_model.interact('transformer')
# run_config = getattr(tuning_model, str(run))

DATASET_NAME = tuning_model.datasets[run_config.run[0]]

metrics = tuning_model.EvaluationMetricsConfiguration(
    run=f"{run_config.run}_transformer_metrics")

print(f"\033[42m\033[30m{DATASET_NAME=}\033[0m")
print(f"\033[43m\033[30m{run_config}\033[0m")


@torch.no_grad()
def estimate_val_loss(model, eval_iters=2):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch("val")
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_model(raw_model, run_config, metrics, filename=".pt"):
    checkpoint = {"model_state_dict": raw_model.state_dict(), }
    Path('results').mkdir(exist_ok=True)
    tuning_model.save_metrics(run_config, metrics)
    model_name = f"{run_config.run}{filename}"
    torch.save(checkpoint, Path('results') / str(model_name))
    print(f"Model saved as {model_name}")


# GPU / precision setup
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


# Training config
BLOCK_SIZE = run_config.train_block_size
BATCH_SIZE = run_config.train_batch_size
MAX_ITERS = run_config.train_max_iters
LEARNING_RATE = run_config.train_learning_rate
WEIGHT_DECAY = run_config.train_weight_decay
LOG_FREQ = run_config.train_log_freq


def get_batch(split):
    # NEW
    if DATASET_NAME == "mixed":
        if np.random.rand() < 0.7:
            data = load_dataset.load_wiki()
            if split == "train":
                data = data[: int(0.9 * len(data))]
            else:
                data = data[int(0.9 * len(data)):]
        else:
            data = load_dataset.load_tinystories()[split]

    elif DATASET_NAME == "tinystories":
        data = load_dataset.load_tinystories()[split]

    elif DATASET_NAME == "wiki":
        data = load_dataset.load_wiki()
        if split == "train":
            data = data[: int(0.9 * len(data))]
        else:
            data = data[int(0.9 * len(data)):]

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([
        torch.from_numpy(data[i: i + BLOCK_SIZE].astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(data[i + 1: i + 1 + BLOCK_SIZE].astype(np.int64))
        for i in ix
    ])

    if torch.cuda.is_available():
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# Training
def main():
    import time
    start_time = time.time()

    model = Transformer(
        n_layer=run_config.tf_n_layer,
        d_model=run_config.tf_d_model,
        n_head=run_config.tf_n_head,
        mlp_mult=run_config.tf_mlp_mult,
        dropout=run_config.tf_dropout,
        vocab_size=run_config.tf_vocab_size,
        max_seq_len=run_config.train_block_size,
    ).to(device)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0

    best_val_loss = float("inf")
    best_model_state = None

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
            avg_train_loss = loss_acc.item() / loss_steps
            val_loss = estimate_val_loss(model, run_config.eval_iters)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())

            print(
                f"Step {step}/{MAX_ITERS} | "
                f"train {avg_train_loss:.3f} | "
                f"val {val_loss:.3f}"
            )

            metrics.steps.append(step)
            metrics.train_loss.append(avg_train_loss)
            metrics.val_loss.append(val_loss)

            loss_acc = 0
            loss_steps = 0

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    metrics.elapsed_time = (time.time() - start_time) / 60
    metrics.perplexity = metrics.calculate_perplexity(metrics.val_loss)

    save_model(model, run_config, metrics)

    print("Training done, now generating a sample")

    model.eval()
    prompt_text = "Gravity\n\nGravity is"

    prompt = torch.tensor(
        bytearray(prompt_text, "utf-8"),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    ret = model.generate(prompt, max_new_tokens=100, top_k=3)

    ret_decoded = bytes(
        ret.to(torch.uint8).to("cpu").squeeze(0)
    ).decode(errors="backslashreplace")

    print(ret_decoded)

    print(f"\033[43m\033[30m{metrics}\033[0m")


if __name__ == "__main__":
    main()
