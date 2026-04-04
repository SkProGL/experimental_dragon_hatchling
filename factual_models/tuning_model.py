import torch.nn.functional as F
from dataclasses import dataclass
import math
import torch
from dataclasses import field  # used for mutable instances
from typing import List
import json
from dataclasses import asdict

# generation


@torch.no_grad()
def top_k_next_tokens(model, device, prompt, k=5):
    idx = torch.tensor(
        list(prompt.encode("utf-8")),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    idx_cond = idx[:, -model.max_seq_len:] if hasattr(model, "max_seq_len") else idx
    logits, _ = model(idx_cond)

    probs = F.softmax(logits[:, -1, :], dim=-1)

    topk_probs, topk_ids = torch.topk(probs, k)

    results = []
    for p, i in zip(topk_probs[0], topk_ids[0]):
        token = bytes([i.item()]).decode("utf-8", errors="ignore")
        results.append((repr(token), p.item()))

    return results


def to_ids(text, device):
    return torch.tensor(
        list(text.encode("utf-8")),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)


@torch.no_grad()
def greedy_decode(model, device, prompt_text, max_new_tokens=12):
    idx = to_ids(prompt_text, device)

    for _ in range(max_new_tokens):
        # Transformer has max_seq_len; BDH in your code does not expose one
        idx_cond = idx[:, -model.max_seq_len:] if hasattr(model, "max_seq_len") else idx

        logits, _ = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # argmax, no sampling
        idx = torch.cat([idx, next_id], dim=1)

    full_text = bytes(idx.squeeze(0).tolist()).decode("utf-8", errors="ignore")
    return full_text[len(prompt_text):]  # continuation only


@torch.no_grad()
def score_candidate(model, device, prompt_text, candidate_text):
    # Important: make prompt end with "A: " so candidate starts cleanly
    prompt_ids = list(prompt_text.encode("utf-8"))
    cand_ids = list(candidate_text.encode("utf-8"))

    full_ids = torch.tensor(
        [prompt_ids + cand_ids],
        dtype=torch.long,
        device=device
    )

    # predict next byte for every position
    logits, _ = model(full_ids[:, :-1])
    log_probs = F.log_softmax(logits[0], dim=-1)

    # candidate bytes start being predicted at position len(prompt_ids)-1
    start = len(prompt_ids) - 1
    target = torch.tensor(cand_ids, dtype=torch.long, device=device)

    token_logps = log_probs[start:start + len(cand_ids)].gather(
        1, target.unsqueeze(1)
    ).squeeze(1)

    return {
        "avg_logprob": token_logps.mean().item(),  # better when answer lengths differ
        "sum_logprob": token_logps.sum().item(),
    }


def rank_candidates(model, device, prompt_text, candidates):
    scored = []
    for cand in candidates:
        s = score_candidate(model, device, prompt_text, cand)
        scored.append((cand, s["avg_logprob"], s["sum_logprob"]))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def seed_prompts(seed=42):
    import random
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_ranking_table(run_config, model, device):
    qs = [
        {
            "q": "The capital of France is",
            "p": "Context: Paris\nQ: The capital of France is\nA: ",
            "c": ["Paris", "Lyon", "London", "Canberra"],
            "gt": "Paris",
        },
        {
            "q": "The chemical symbol for gold is",
            "p": "Context: Au\nQ: The chemical symbol for gold is\nA: ",
            "c": ["Au", "Ag", "Fe", "Cu"],
            "gt": "Au",
        },
        {
            "q": "The square root of 144 is",
            "p": "Context: 12\nQ: The square root of 144 is\nA: ",
            "c": ["12", "14", "10", "16"],
            "gt": "12",
        },
    ]

    lines = ["\n## Model preference over candidate answers"]
    lines.append("| Question | Candidates | Ranking (average log-probability) | Ground truth |")
    lines.append("|----------|------------|------------------------|--------------|")

    for item in qs:
        r = rank_candidates(model, device, item["p"], item["c"])
        r_txt = ", ".join([f"{x} ({s:.2f})" for x, s, _ in r])

        line = f"| {item['q']} | {', '.join(item['c'])} | {r_txt} | {item['gt']} |"
        lines.append(line)   # also saves

    for l in lines:
        print(l)          # still prints

    from pathlib import Path
    with open(Path(__file__).parent / "inference" / f"{run_config.run}_questions.md", "a") as f:
        f.write("\n".join(lines) + "\n\n")


def print_greedy_table(run_config, model, device):
    qs = [
        ("The capital of France is", "The capital of France is"),
        ("Q: The capital of France is A:", "Q: The capital of France is A: "),
        ("Context: Paris Q: The capital of France is A:", "Context: Paris Q: The capital of France is A: "),
    ]

    def format_topk(prompt):
        tk = top_k_next_tokens(model, device, prompt, k=4)

        parts = []
        for tok, prob in tk:
            # tok = tok.strip("'")
            if tok == "\n":
                tok = "\\n"
            elif tok == " ":
                tok = "[space]"
            parts.append(f"{tok} ({prob:.2f})")

        return ", ".join(parts)

    content = ["## Greedy decoding outputs and top-4 next-token predictions"]

    content.append("| Prompt | Greedy output | Top tokens |")
    content.append("|--------|---------------|------------|")

    for label, prompt in qs:
        out = greedy_decode(model, device, prompt, max_new_tokens=8)
        out = out.replace("\n", "\\n").replace("|", "\\|")

        row = f"| {label} | `{out}` | {format_topk(prompt)} |"

        content.append(row)

    for r in content:
        print(r)
    from pathlib import Path
    with open(Path(__file__).parent / "inference" / f"{run_config.run}_questions.md", "a") as f:
        f.write("\n".join(content) + "\n\n")
# def print_greedy_table(run_config, model, device):
#     qs = [
#         ("The capital of France is", "The capital of France is"),
#         ("Q: The capital of France is A:", "Q: The capital of France is A: "),
#         ("Context: Paris Q: The capital of France is A:", "Context: Paris Q: The capital of France is A: "),
#     ]
#
#     def format_topk(prompt):
#         tk = top_k_next_tokens(model, device, prompt, k=4)
#
#         parts = []
#         for tok, prob in tk:
#             # tok = tok.strip("'")
#             if tok == "\n":
#                 tok = "\\n"
#             elif tok == " ":
#                 tok = "[space]"
#             parts.append(f"{tok} ({prob:.2f})")
#
#         return ", ".join(parts)
#
#     print("| Prompt | Greedy output | Top tokens |")
#     print("|--------|---------------|------------|")
#
#     for label, prompt in qs:
#         out = greedy_decode(model, device, prompt, max_new_tokens=8)
#         out = out.replace("\n", "\\n").replace("|", "\\|")
#
#         print(f"| {label} | `{out}` | {format_topk(prompt)} |")
#
#     from pathlib import Path
#     with open(Path(__file__).parent / "inference" / f"{run_config.run}_questions.md", "a") as f:
#         f.write("\n".join(content) + "\n\n")


def extra_questions(model, device):
    # prompt = "Context: Paris\nQ: The capital of France is\nA: "
    prompt = "Q: The capital of France is\nA: "
    # prompt = "The capital of France is "
    print(prompt)
    out = greedy_decode(model, device, prompt, max_new_tokens=8)

    print(f"\033[43m\033[30mgreedy_decode\033[0m")
    print(repr(out))

    prompt = "Context: Paris\nQ: The capital of France is\nA: "
    # prompt = "The capital of France is "
    candidates = ["Paris", "Lyon", "London", "Canberra"]
    print(f"\033[43m\033[30mrand_candidates\033[0m")
    print(rank_candidates(model, device, prompt, candidates))

    print(f"\033[43m\033[30mtop_k\033[0m")
    print(top_k_next_tokens(model, device, prompt))


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
    print("\nNORMAL QUESTIONS")
    normal_table = format_rows(normal_rows)

    print("\nCORRUPTED QUESTIONS")
    corrupted_table = format_rows(corrupted_rows)
    print("\nCONTEXT QUESTIONS")
    context_table = format_rows(context_rows)
    # save to file
    with open(Path(__file__).parent / "inference" / f"{run_config.run}_questions.md", "w") as f:
        f.write("## Normal questions\n")
        f.write("\n| Prompt | Output |\n|--------|--------|\n")
        f.write("\n".join(normal_table))

        f.write("\n\n## Corrupted questions\n")
        f.write("\n| Prompt | Output |\n|--------|--------|\n")
        f.write("\n".join(corrupted_table))

        f.write("\n\n## Context questions\n")
        f.write("\n| Prompt | Output |\n|--------|--------|\n")
        f.write("\n".join(context_table))


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
        prefixes = ["B", "D", "F"]
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
