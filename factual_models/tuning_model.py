from dataclasses import dataclass
import math
import torch
from dataclasses import field  # used for mutable instances
from typing import List
import json
from dataclasses import asdict

# generation


def generate_text(device, model, prompt_text, max_new_tokens=100, top_k=3, temperature=1.0):
    prompt = torch.tensor(
        bytearray(prompt_text, "utf-8"),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            # temperature=temperature,
        )

    decoded = bytes(
        output.to(torch.uint8).cpu().squeeze(0)
    ).decode(errors="backslashreplace")

    return decoded


def run_questions_from_file(run_config, device, filepath, model):
    from pathlib import Path

    normal_rows = []
    corrupted_rows = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    section = None  # "normal" or "corrupted"
    context_rows = []
    answers = []

    for line in lines:
        line = line.strip()

        if line == "Questions":
            section = "normal"
            continue

        if line == "Corrupted questions":
            section = "corrupted"
            continue

        if line == "Answers":
            section = "answers"
            continue

        # collect answers
        if section == "answers" and line:
            answer = line.split(".", 1)[-1].strip()
            answers.append(answer)
            continue

        if section and line:
            question = line.split(".", 1)[-1].strip()
            result = generate_text(device, model, question)

            if section == "normal":
                normal_rows.append((question, result))
            elif section == "corrupted":
                corrupted_rows.append((question, result))
    # build context prompts (only for normal questions)
    for i, (question, _) in enumerate(normal_rows):
        if i < len(answers):
            context = answers[i]
            prompt = f"Context: {context}\nQ: {question}\nA:"
            result = generate_text(device, model, prompt)
            context_rows.append((prompt, result))

    def format_rows(rows):
        formatted = []
        for i, (prompt, output) in enumerate(rows, 1):
            safe_prompt = prompt.replace("|", "\\|").replace("\n", " ")
            safe_output = output.replace("|", "\\|").replace("\n", " ")

            line = f"| {i}. {safe_prompt} | {safe_output[len(safe_prompt):]} |"
            print(line)
            formatted.append(line)
        return formatted

    # print both tables
    print("\n=== NORMAL QUESTIONS ===")
    normal_table = format_rows(normal_rows)

    print("\n=== CORRUPTED QUESTIONS ===")
    corrupted_table = format_rows(corrupted_rows)
    print("\n=== CONTEXT QUESTIONS ===")
    context_table = format_rows(context_rows)
    # save to file
    with open(Path(__file__).parent / "inference" / f"{run_config.run}_questions.md", "w") as f:
        f.write("## Normal Questions\n")
        f.write("\n| Prompt | Output |\n|--------|--------|\n")
        f.write("\n".join(normal_table))

        f.write("\n\n## Corrupted Questions\n")
        f.write("\n| Prompt | Output |\n|--------|--------|\n")
        f.write("\n".join(corrupted_table))

        # f.write("\n\n## Context Questions\n")
        # f.write("\n| Prompt | Output |\n|--------|--------|\n")
        # f.write("\n".join(context_table))


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
        prefixes = ["A", "C", "E"]
        cpu_names = [f"{p}cpu" for p in prefixes]
        runs = [
            globals()[name]
            for name in sorted(globals())
            if name in cpu_names or any(name.startswith(prefix) and name[1:].isdigit() for prefix in prefixes)
        ]

        for i, r in enumerate(runs):
            print(
                f"{i}: {r.run}, L={r.bdh_n_layer}, D={r.bdh_n_embd}, H={r.bdh_n_head}, BATCH_SIZE={r.train_batch_size}")

    elif model_type == "transformer":
        prefixes = ["B", "D", "E"]
        cpu_names = [f"{p}cpu" for p in prefixes]
        runs = [
            globals()[name]
            for name in sorted(globals())
            if name in cpu_names or any(name.startswith(prefix) and name[1:].isdigit() for prefix in prefixes)
        ]

        for i, r in enumerate(runs):
            print(
                f"{i}: {r.run}, L={r.tf_n_layer}, D={r.tf_d_model}, H={r.tf_n_head}, BATCH_SIZE={r.train_batch_size}")

    else:
        raise ValueError("model_type must be 'bdh' or 'transformer'")

    try:
        r = runs[int(input("Select: "))]
        size = input("Batch size: ")
        r.train_batch_size = int(size) if size else r.train_batch_size
        return r
    except:
        return runs[0]


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
    model_type: str = "dragon_hatchling"


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
    model_type: str = "transformer"


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


def produce_dragon(prefix, spec):
    r, l, e, h, m, lr, bs, mi, wd = spec
    return DragonConfiguration(
        run=prefix+r,
        bdh_n_layer=l,
        bdh_n_embd=e,
        bdh_n_head=h,
        bdh_mlp_internal_dim_multiplier=m,
        train_learning_rate=lr,
        train_batch_size=bs,
        train_max_iters=mi,
        train_weight_decay=wd,
    )


def produce_transformer(prefix, spec):
    r, l, d, h, m, lr, bs, mi, wd = spec
    return TransformerConfiguration(
        run=prefix+r,
        tf_n_layer=l,
        tf_d_model=d,
        tf_n_head=h,
        tf_mlp_mult=m,
        train_learning_rate=lr,
        train_batch_size=bs,
        train_max_iters=mi,
        train_weight_decay=wd,
    )


def produce_globals(hyperparams, run, run_type):
    for spec in hyperparams:
        if run_type == "bdh":
            globals()[f"{run}{spec[0]}"] = produce_dragon(run, spec)
        elif run_type == "tf":
            globals()[f"{run}{spec[0]}"] = produce_transformer(run, spec)


hyperparams = [
    ["cpu", 6, 256, 4, 32, 1e-3, 8, 1, 0.1],
    ["1", 6, 256, 4, 64, 1e-3, 8, 6000, 0.1],
    ["2", 8, 384, 6, 64, 5e-4, 4, 12000, 0.1],
    ["3", 12, 512, 8, 64, 3e-4, 2, 20000, 0.1],
    ["4", 8, 384, 6, 128, 5e-4, 2, 12000, 0.05],
    ["5", 8, 384, 8, 128, 5e-4, 2, 12000, 0.05],
    ["6", 8, 512, 8, 128, 4e-4, 2, 12000, 0.05],
    ["7", 8, 512, 16, 128, 4e-4, 2, 12000, 0.05],
    ["8", 8, 512, 8, 256, 4e-4, 1, 12000, 0.01],
    ["9", 8, 512, 8, 256, 4e-4, 1, 12000, 0.1],
    ["10", 8, 384, 6, 64, 5e-4, 4, 30000, 0.1],
]

mix_hyperparams = [
    ["cpu", 6, 256, 4, 32, 1e-3, 8, 1, 0.1],
    ["2", 8, 384, 6, 64, 5e-4, 4, 12000, 0.1],
    ["10", 8, 384, 6, 64, 5e-4, 4, 30000, 0.1],
]
datasets = {"A": "wiki",
            "B": "wiki",
            "C": "tinystories",
            "D": "tinystories",
            "E": "mixed",
            "F": "mixed"}


produce_globals(hyperparams, 'A', 'bdh')
produce_globals(hyperparams, 'B', 'tf')
produce_globals(hyperparams, 'C', 'bdh')
produce_globals(hyperparams, 'D', 'tf')
produce_globals(mix_hyperparams, 'E', 'bdh')
produce_globals(mix_hyperparams, 'F', 'tf')
