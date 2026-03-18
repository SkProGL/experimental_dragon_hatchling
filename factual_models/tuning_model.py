from dataclasses import dataclass
import math
from dataclasses import field  # used for mutable instances
from typing import List
import json
from dataclasses import asdict


def save_metrics(run_config, metrics):
    filepath = f"results/{run_config.run}.json"
    combined = {"config": asdict(run_config), "metrics": asdict(metrics), }

    with open(filepath, "w") as f:
        json.dump(combined, f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(run_name):
    filepath = f"{run_name}.json"
    with open(filepath, "r") as f:
        data = json.load(f)
    print(data)
    return DragonConfiguration(**data)


def interact(model_type="bdh"):
    if model_type == "bdh":
        runs = [ACPU] + [globals()[f"A{i}"] for i in range(1, 11)]

        for i, r in enumerate(runs):
            print(
                f"{i}: {r.run}, L={r.bdh_n_layer}, D={r.bdh_n_embd}, H={r.bdh_n_head}, BATCH_SIZE={r.train_batch_size}")

    elif model_type == "transformer":
        runs = [BCPU, B1]  # extend later if needed

        for i, r in enumerate(runs):
            print(
                f"{i}: {r.run}, L={r.tf_n_layer}, D={r.tf_d_model}, H={r.tf_n_head}, BATCH_SIZE={r.train_batch_size}")

    else:
        raise ValueError("model_type must be 'bdh' or 'transformer'")

    try:
        r = runs[int(input("Select: "))]
        r.train_batch_size = int(input("Batch size: "))
        return r
    except:
        return runs[0]
# def interact():
#     runs = [CPU] + [globals()[f"A{i}"] for i in range(1, 11)]
#     for i, r in enumerate(runs):
#         print(f"{i}: {r.run}, L={r.bdh_n_layer}, D={r.bdh_n_embd}, H={r.bdh_n_head}, BATCH_SIZE={r.train_batch_size}")
#     try:
#         r = runs[int(input("Select: "))]
#         r.train_batch_size = (int(input("Batch size: ")))
#         return r
#     except:
#         return CPU


@dataclass
class DragonConfiguration:
    # bdh.py specific
    run: str
    bdh_n_layer: int = 6
    bdh_n_embd: int = 256
    bdh_dropout: float = 0.1
    bdh_n_head: int = 4
    bdh_mlp_internal_dim_multiplier: int = 128
    bdh_vocab_size: int = 256

    # train.py specific
    train_block_size: int = 512
    train_batch_size: int = 8
    train_max_iters: int = 3000
    train_learning_rate: float = 1e-3
    train_weight_decay: float = 0.1
    train_log_freq: int = 100

    eval_iters: int = 10


@dataclass
class TransformerConfiguration:
    # transformer-specific
    run: str
    tf_n_layer: int = 6
    tf_d_model: int = 256
    tf_dropout: float = 0.1
    tf_n_head: int = 4
    tf_mlp_mult: int = 64
    tf_vocab_size: int = 256

    # train.py specific
    train_block_size: int = 512
    train_batch_size: int = 8
    train_max_iters: int = 3000
    train_learning_rate: float = 1e-3
    train_weight_decay: float = 0.1
    train_log_freq: int = 100

    eval_iters: int = 10


@dataclass
class EvaluationMetricsConfiguration:
    run: str
    steps: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    perplexity: List[float] = field(default_factory=list)
    elapsed_time: float | None = None
    latent_dim_per_head: int | None = None
    # latent dimension inside one layer for one token 16,384
    total_latent_per_layer: int | None = None
    # calculated as sparsity = (x_sparse == 0).float().mean()
    sparsity_ratio: float | None = None

    def calculate_perplexity(self, val_loss):
        return [math.exp(v) for v in val_loss]


ACPU = DragonConfiguration(
    run="cpu",
    bdh_n_layer=6,
    bdh_n_embd=256,
    bdh_n_head=4,
    bdh_mlp_internal_dim_multiplier=32,
    # use default values for: dropout, vocabulary size
    train_block_size=512,
    train_learning_rate=1e-3,
    train_batch_size=8,
    train_max_iters=1,
    train_weight_decay=0.1,
    eval_iters=1,
    train_log_freq=1,
)
A1 = DragonConfiguration(
    run="A1",
    bdh_n_layer=6,
    bdh_n_embd=256,
    bdh_n_head=4,
    bdh_mlp_internal_dim_multiplier=64,
    # use default values for: dropout, vocabulary size
    train_learning_rate=1e-3,
    train_batch_size=16,
    train_max_iters=6000,
    train_weight_decay=0.1,
    # use default values for: block size, log frequency
)

A2 = DragonConfiguration(
    run="A2",
    bdh_n_layer=8,
    bdh_n_embd=384,
    bdh_n_head=6,
    bdh_mlp_internal_dim_multiplier=64,
    train_learning_rate=5e-4,
    train_batch_size=16,
    train_max_iters=12000,
    train_weight_decay=0.1,
)

A3 = DragonConfiguration(
    run="A3",
    bdh_n_layer=12,
    bdh_n_embd=512,
    bdh_n_head=8,
    bdh_mlp_internal_dim_multiplier=64,
    train_learning_rate=3e-4,
    train_batch_size=16,
    train_max_iters=20000,
    train_weight_decay=0.1,
)

A4 = DragonConfiguration(
    run="A4",
    bdh_n_layer=8,
    bdh_n_embd=384,
    bdh_n_head=6,
    bdh_mlp_internal_dim_multiplier=128,
    train_learning_rate=5e-4,
    train_batch_size=16,
    train_max_iters=12000,
    train_weight_decay=0.05,
)

A5 = DragonConfiguration(
    run="A5",
    bdh_n_layer=8,
    bdh_n_embd=384,
    bdh_n_head=8,
    bdh_mlp_internal_dim_multiplier=128,
    train_learning_rate=5e-4,
    train_batch_size=16,
    train_max_iters=12000,
    train_weight_decay=0.05,
)

A6 = DragonConfiguration(
    run="A6",
    bdh_n_layer=8,
    bdh_n_embd=512,
    bdh_n_head=8,
    bdh_mlp_internal_dim_multiplier=128,
    train_learning_rate=4e-4,
    train_batch_size=16,
    train_max_iters=12000,
    train_weight_decay=0.05,
)

A7 = DragonConfiguration(
    run="A7",
    bdh_n_layer=8,
    bdh_n_embd=512,
    bdh_n_head=10,
    bdh_mlp_internal_dim_multiplier=128,
    train_learning_rate=4e-4,
    train_batch_size=16,
    train_max_iters=12000,
    train_weight_decay=0.05,
)

A8 = DragonConfiguration(
    run="A8",
    bdh_n_layer=8,
    bdh_n_embd=512,
    bdh_n_head=8,
    bdh_mlp_internal_dim_multiplier=256,
    train_learning_rate=4e-4,
    train_batch_size=16,
    train_max_iters=12000,
    train_weight_decay=0.01,
)

A9 = DragonConfiguration(
    run="A9",
    bdh_n_layer=8,
    bdh_n_embd=512,
    bdh_n_head=8,
    bdh_mlp_internal_dim_multiplier=256,
    train_learning_rate=4e-4,
    train_batch_size=16,
    train_max_iters=12000,
    train_weight_decay=0.1,
)

A10 = DragonConfiguration(
    run="A10",
    bdh_n_layer=8,
    bdh_n_embd=384,
    bdh_n_head=6,
    bdh_mlp_internal_dim_multiplier=64,
    train_learning_rate=5e-4,
    train_batch_size=4,
    train_max_iters=30000,
    train_weight_decay=0.1,
)


BCPU = TransformerConfiguration(
    run="cpu",
    tf_n_layer=6,
    tf_d_model=256,
    tf_n_head=4,
    tf_mlp_mult=64,
    train_learning_rate=1e-3,
    train_batch_size=8,
    train_max_iters=1,
    train_weight_decay=0.1,
)
B1 = TransformerConfiguration(
    run="B1",
    tf_n_layer=6,
    tf_d_model=256,
    tf_n_head=4,
    tf_mlp_mult=64,
    train_learning_rate=1e-3,
    train_batch_size=16,
    train_max_iters=6000,
    train_weight_decay=0.1,
)
