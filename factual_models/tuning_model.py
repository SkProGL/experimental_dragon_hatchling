from dataclasses import dataclass
import math
from dataclasses import field  # used for mutable instances
from typing import List
import json
import os
from dataclasses import asdict

# Run	Layers	Emb	Heads	MLP Mult	LR	Batch	Weight Decay	Iterations
# A1	6	256	4	64	1e-3	16	0.1	6k


def save_metrics(metrics):
    filepath = f"{metrics.run}.json"
    with open(filepath, "w") as f:
        json.dump(asdict(metrics), f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(run_name):
    filepath = f"{run_name}.json"
    with open(filepath, "r") as f:
        data = json.load(f)
    print(data)
    return RunConfiguration(**data)


@dataclass
class RunConfiguration:
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


CPU = RunConfiguration(
    run="cpu",
    bdh_n_layer=6,
    bdh_n_embd=256,
    bdh_n_head=4,
    bdh_mlp_internal_dim_multiplier=32,
    # use default values for: dropout, vocabulary size
    train_block_size=512,
    train_learning_rate=1e-3,
    train_batch_size=8,
    train_max_iters=3,
    train_weight_decay=0.1,
    eval_iters=5,
    train_log_freq=1,
)
A1 = RunConfiguration(
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

A2 = RunConfiguration(
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

A3 = RunConfiguration(
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


@dataclass
class EvaluationMetricsConfiguration:
    run: str
    steps: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    perplexity: List[float] = field(default_factory=list)
    elapsed_time: float | None = None
    # optional
    # latent dimension per head 4096
    latent_dim_per_head: int | None = None
    # latent dimension inside one layer for one token 16,384
    total_latent_per_layer: int | None = None
    # calculated as sparsity = (x_sparse == 0).float().mean()
    sparsity_ratio: float | None = None

    def calculate_perplexity(self, val_loss):
        return [math.exp(v) for v in val_loss]


# metrics_A1 = EvaluationMetricsConfiguration(
#     run="A1",
#     steps=[0, 100, 200],
#     train_loss=[3.1, 2.8, 2.6],
#     val_loss=[3.2, 2.9, 2.7],
#     perplexity=[22.1, 18.3, 16.4],
#     elapsed_time=1.75,
# )
save_metrics(CPU)
load_metrics('cpu')
