# Copyright Pathway Technology, Inc.
from template.gpu_support import GPUSupport
from factual_models import wiki_bdh as bdh
from factual_models import tuning_model
import math
import copy
import requests
import numpy as np
from contextlib import nullcontext
import os
import torch
import pandas as pd

# SPECIFY HERE THE RUN CONFIG (e.g. A1, A2)
run_config = tuning_model.CPU
metrics = tuning_model.EvaluationMetricsConfiguration(
    run=f"{run_config.run}_metrics")
print(f"\033[43m\033[30m{run_config}\033[0m")


def calculate_latent_dim() -> tuning_model.EvaluationMetricsConfiguration:
    """Calculates latent dimensions"""
    D = run_config.bdh_n_embd
    nh = run_config.bdh_n_head
    mlp = run_config.bdh_mlp_internal_dim_multiplier

    N = D * mlp // nh

    return N, nh*N


@torch.no_grad()
def estimate_val_loss(model, eval_iters=2):
    """Validation loss calculation"""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch("val")
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_model(raw_model, configs=[], filename="wiki_model.pt"):
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
    }
    for i in configs:
        tuning_model.save_metrics(i)

    torch.save(checkpoint, filename)
    print(f"Model saved as {filename}")


GPUSupport()
metrics.latent_dim_per_head, metrics.total_latent_per_layer = calculate_latent_dim()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
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
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# configuration
BDH_CONFIG = bdh.BDHConfig(
    n_layer=run_config.bdh_n_layer,
    n_embd=run_config.bdh_n_embd,
    dropout=run_config.bdh_dropout,
    n_head=run_config.bdh_n_head,
    mlp_internal_dim_multiplier=run_config.bdh_mlp_internal_dim_multiplier,
    vocab_size=run_config.bdh_vocab_size
)
BLOCK_SIZE = run_config.train_block_size
BATCH_SIZE = run_config.train_batch_size
MAX_ITERS = run_config.train_max_iters
LEARNING_RATE = run_config.train_learning_rate
WEIGHT_DECAY = run_config.train_weight_decay
LOG_FREQ = run_config.train_log_freq

# [old] tinyshakespeare
# input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

WIKI_PATH = os.path.join(os.path.dirname(__file__), "simple-wikipedia.parquet")
_wiki_bytes = None

# [old] tinyshakespeare
# fetch the tiny shakespeare dataset
# def fetch_data():
#     if not os.path.exists(input_file_path):
#         data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#         with open(input_file_path, "w") as f:
#             f.write(requests.get(data_url).text)


def load_wiki_bytes():
    global _wiki_bytes
    if _wiki_bytes is not None:
        return _wiki_bytes

    df = pd.read_parquet(WIKI_PATH)

    texts = df["text"].astype(str).tolist()

    # marks document boundaries, helps model learn
    full_text = "\n\n<doc>\n\n".join(texts)

    # converts string into utf-8 encoded raw bytes,
    # then converts to byte-level integer array (0-255)
    _wiki_bytes = np.frombuffer(
        full_text.encode("utf-8"),
        dtype=np.uint8,
    )
    return _wiki_bytes


def get_batch(split):
    # [old] tinyshakespeare
    # data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    # [new] wiki
    # treat the file as bytes
    data = load_wiki_bytes()

    # training and validation split .9 to .1
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)):]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i: i + BLOCK_SIZE]).astype(np.int64))
         for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (data[i + 1: i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


def main():
    # fetch_data()
    import time
    start_time = time.time()

    model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0

    best_val_loss = float("inf")
    best_model_state = None
    for step in range(MAX_ITERS):
        with ctx:

            if step % LOG_FREQ == 0:
                logits, loss, sparsity = model(x, y, return_sparsity=True)
                metrics.sparsity_ratio = sparsity.item()
            else:
                logits, loss = model(x, y)
        x, y = get_batch("train")
        loss_acc += loss
        loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % LOG_FREQ == 0:
            # output metrics related calculations
            avg_train_loss = loss_acc.item() / loss_steps
            val_loss = estimate_val_loss(model, run_config.eval_iters)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
            # print(
            #     f"Step: {step}/{MAX_ITERS} loss {avg_train_loss:.3}")
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
    save_model(model, [run_config, metrics])
    print("Training done, now generating a sample ")
    model.eval()
    prompt_text = "Gravity\n\nGravity is"
    prompt = torch.tensor(
        # bytearray("To be or ", "utf-8"),
        bytearray(prompt_text, "utf-8"),
        dtype=torch.long, device=device
    ).unsqueeze(0)

    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)

    print(f"\033[43m\033[30m{metrics}\033[0m")
