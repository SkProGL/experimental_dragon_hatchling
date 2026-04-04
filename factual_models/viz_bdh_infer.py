import os
import math
import shutil
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from template.gpu_support import GPUSupport
from factual_models import bdh_model
from factual_models import tuning_model


print("Available model files:")
model_dir = Path(__file__).parent / "models"
for path in sorted(model_dir.iterdir()):
    if path.is_file():
        print(path.name)

# select run config
run_config = tuning_model.interact("bdh")
DATASET_NAME = tuning_model.datasets[run_config.run[0]]

MODEL_PATH = Path(__file__).parent / "models" / f"{run_config.run}.pt"

print(f"\033[42m\033[30m{DATASET_NAME=}\033[0m")
print(f"\033[43m\033[30m{run_config}\033[0m")

GPUSupport()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model loading
def load_model():
    config = bdh_model.BDHConfig(
        n_layer=run_config.bdh_n_layer,
        n_embd=run_config.bdh_n_embd,
        dropout=run_config.bdh_dropout,
        n_head=run_config.bdh_n_head,
        mlp_internal_dim_multiplier=run_config.bdh_mlp_internal_dim_multiplier,
        vocab_size=run_config.bdh_vocab_size,
    )

    model = bdh_model.BDH(config).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)["model_state_dict"]

    # fix: remove "_orig_mod." prefix if present
    new_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict)

    model.eval()
    return model


# token helpers
def text_to_ids(text, device):
    return torch.tensor(
        list(text.encode("utf-8")),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)


def ids_to_text(idx):
    return bytes(idx.squeeze(0).tolist()).decode("utf-8", errors="ignore")


# latent extraction
@torch.no_grad()
def collect_bdh_latents(model, idx):
    """
    Replays BDH forward pass and captures x_sparse activations per layer.

    Returns:
        dict with:
            "latents": numpy array [layers, tokens, latent_dim]
            "tokens": list[str]
            "token_count": int
            "layer_count": int
            "latent_dim": int
    """
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.embed(idx).unsqueeze(1)   # [B, 1, T, D]
    x = model.ln(x)

    layer_latents = []

    for level in range(C.n_layer):
        x_latent = x @ model.encoder          # [B, nh, T, N]
        x_sparse = F.relu(x_latent)           # [B, nh, T, N]

        # collapse heads -> one vector per token for this layer
        token_latents = x_sparse.mean(dim=1).squeeze(0)   # [T, N]
        layer_latents.append(token_latents.detach().cpu().numpy())

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)

        yMLP = (
            xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
        )  # [B,1,T,D]

        y = model.ln(yMLP)
        x = model.ln(x + y)

    tokens = []
    for i in idx.squeeze(0).tolist():
        tok = bytes([i]).decode("utf-8", errors="ignore")
        if tok == "\n":
            tok = "\\n"
        elif tok == "\t":
            tok = "\\t"
        elif tok == " ":
            tok = "␠"
        elif tok == "":
            tok = "?"
        tokens.append(tok)

    latents = np.stack(layer_latents, axis=0)   # [L, T, N]

    return {
        "latents": latents,
        "tokens": tokens,
        "token_count": T,
        "layer_count": C.n_layer,
        "latent_dim": N,
    }


# ANIMATION CORE
def init_positions(num_latents, seed=1):
    """
    Fast deterministic radial-ish initialization.
    Much lighter than spring layout and scales better for bigger N.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, num_latents, endpoint=False)
    radii = 1.0 + 0.08 * rng.standard_normal(num_latents)

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    pos = np.stack([x, y], axis=1)
    pos += rng.normal(scale=0.03, size=pos.shape)
    return pos.copy(), pos.copy()


def compute_sparse_corr(A, t, min_steps=2, k=6):
    """
    A: [latents, time]
    """
    if t < min_steps:
        return None, None, None

    A_t = A[:, :t + 1]

    X = A_t - A_t.mean(axis=1, keepdims=True)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    n = X.shape[0]
    k = min(k, max(1, n - 1))

    neigh_idx = np.empty((n, k), dtype=np.int32)
    neigh_w = np.empty((n, k), dtype=np.float32)
    strength = np.zeros(n, dtype=np.float32)

    block = 512
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = X[start:end] @ X.T
        sims = np.nan_to_num(sims)
        sims = np.clip(sims, 0.0, None)

        row_ids = np.arange(start, end)
        sims[np.arange(end - start), row_ids] = 0.0

        idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(sims, idx, axis=1)

        order = np.argsort(vals, axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)

        neigh_idx[start:end] = idx
        neigh_w[start:end] = vals
        strength[start:end] = vals.sum(axis=1)

    confidence = min(1.0, (t + 1) / 8.0)
    neigh_w *= confidence
    strength *= confidence

    return neigh_idx, neigh_w, strength


def update_positions_fast(
    pos_dynamic,
    base_pos,
    neigh_idx,
    neigh_w,
    strength,
    t,
    attraction_scale=0.05,
    anchor_scale=0.02,
    center_pull=0.015,
    max_step=0.04,
):
    if neigh_idx is None:
        return pos_dynamic

    pos = pos_dynamic.copy()
    disp = np.zeros_like(pos)

    n, k = neigh_idx.shape

    for i in range(n):
        js = neigh_idx[i]
        ws = neigh_w[i]

        delta = pos[js] - pos[i]
        dist = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-8
        direction = delta / dist

        force = (attraction_scale * (ws[:, None] ** 2)) * direction
        disp[i] += force.sum(axis=0)

    disp += anchor_scale * (base_pos - pos)

    if strength is not None and strength.max() > 0:
        s = strength / (strength.max() + 1e-8)
        disp += (-center_pull * s[:, None]) * pos

    norms = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-8
    scale = np.minimum(1.0, max_step / norms)
    disp *= scale

    confidence = min(1.0, (t + 1) / 8.0)
    pos = pos + confidence * disp
    return pos


def prepare_layer_animation_states(latents, seed=1, k=6):
    """
    latents: [layers, tokens, latent_dim]

    Returns per-layer evolving positions and neighbors for every token step.
    """
    layers, steps, latent_dim = latents.shape
    states = []

    for layer_idx in range(layers):
        A = latents[layer_idx].T  # [latent_dim, time]

        pos_dynamic, base_pos = init_positions(latent_dim, seed=seed + layer_idx)

        layer_steps = []
        neigh_idx = None
        neigh_w = None
        strength = None

        for t in range(steps):
            neigh_idx, neigh_w, strength = compute_sparse_corr(A, t, k=k)

            pos_dynamic = update_positions_fast(
                pos_dynamic=pos_dynamic,
                base_pos=base_pos,
                neigh_idx=neigh_idx,
                neigh_w=neigh_w,
                strength=strength,
                t=t,
            )

            layer_steps.append({
                "pos": pos_dynamic.copy(),
                "A": A,
                "neigh_idx": neigh_idx,
                "neigh_w": neigh_w,
                "strength": strength,
            })

        states.append(layer_steps)

    return states


def layer_grid(n_layers):
    cols = math.ceil(math.sqrt(n_layers))
    rows = math.ceil(n_layers / cols)
    return rows, cols


def plot_layer_frame(ax, pos, A, neigh_idx, neigh_w, t, layer_idx, max_plot_nodes=1200, edge_threshold=0.20,):
    current_act = A[:, t]
    n = len(current_act)

    if n > max_plot_nodes:
        idx_plot = np.linspace(0, n - 1, max_plot_nodes).astype(int)
    else:
        idx_plot = np.arange(n)

    idx_set = set(idx_plot.tolist())

    if neigh_idx is not None:
        for i in idx_plot:
            for j, w in zip(neigh_idx[i], neigh_w[i]):
                if w <= edge_threshold or j not in idx_set or j <= i:
                    continue
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    linewidth=0.25 + 1.1 * float(w),
                    alpha=0.03 + 0.18 * float(w),
                    color="black",
                    zorder=1,
                )

    sizes = 6 + 34 * current_act[idx_plot]
    sc = ax.scatter(
        pos[idx_plot, 0],
        pos[idx_plot, 1],
        s=sizes,
        c=current_act[idx_plot],
        cmap="viridis",
        edgecolors="none",
        zorder=2,
    )

    ax.set_title(f"Layer {layer_idx + 1}", fontsize=14)
    ax.axis("off")
    return sc


def save_composite_frame(
    states,
    tokens,
    t,
    save_path,
    prompt_text,
):
    n_layers = len(states)
    rows, cols = layer_grid(n_layers)

    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    token_display = tokens[t] if t < len(tokens) else "?"
    prompt_preview = prompt_text.replace("\n", "\\n")
    if len(prompt_preview) > 100:
        prompt_preview = prompt_preview[:100] + "..."

    fig.suptitle(
        f"BDH latent activations | token {t + 1}/{len(tokens)} | current token: {repr(token_display)}\n"
        f"Prompt: {prompt_preview}",
        fontsize=16,          # Increased from 12
        fontweight='bold',    # Optional: makes it easier to read in a GIF
    )

    for layer_idx in range(n_layers):
        layer_state = states[layer_idx][t]
        sc = plot_layer_frame(
            ax=axes[layer_idx],
            pos=layer_state["pos"],
            A=layer_state["A"],
            neigh_idx=layer_state["neigh_idx"],
            neigh_w=layer_state["neigh_w"],
            t=t,
            layer_idx=layer_idx,
        )

    for i in range(n_layers, len(axes)):
        axes[i].axis("off")

    # apply tight_layout first, but stop at 90% width (0.9) to leave space on the right
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # add a new custom axis just for the colorbar
    # [left_position, bottom_position, width, height] in fractions of figure size
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    # draw the colorbar into that specific axis
    fig.colorbar(sc, cax=cbar_ax, label="Activation")
    # fig.colorbar(sc, ax=axes.tolist(), shrink=0.6, label="Activation")
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()


def build_gif(frame_paths, output="animation.gif", fps=3):
    images = [imageio.v2.imread(p) for p in frame_paths]
    imageio.mimsave(output, images, fps=fps)


def ensure_clean_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(text, max_len=60):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch == " ":
            keep.append("_")
    out = "".join(keep).strip("_")
    if not out:
        out = "prompt"
    return out[:max_len]


def create_prompt_gif(
    model,
    prompt_text,
    out_dir="animations",
    fps=3,
    k=6,
):
    """
    Main callable:
    - tokenizes prompt
    - extracts layer latents for all prompt tokens
    - builds one frame per token
    - saves GIF
    """
    idx = text_to_ids(prompt_text, device)
    latent_data = collect_bdh_latents(model, idx)

    latents = latent_data["latents"]
    tokens = latent_data["tokens"]

    save_root = ensure_clean_dir(Path(out_dir) / sanitize_filename(prompt_text))
    frames_dir = ensure_clean_dir(save_root / "frames")

    states = prepare_layer_animation_states(latents, k=k)

    frame_paths = []
    for t in range(latent_data["token_count"]):
        frame_path = frames_dir / f"frame_{t:03d}.png"
        save_composite_frame(
            states=states,
            tokens=tokens,
            t=t,
            save_path=frame_path,
            prompt_text=prompt_text,
        )
        frame_paths.append(str(frame_path))

    gif_path = save_root / f"{run_config.run}_latent_animation.gif"
    build_gif(frame_paths, output=str(gif_path), fps=fps)

    print(f"\nSaved GIF: {gif_path}")
    print(
        f"Prompt tokens: {latent_data['token_count']} | "
        f"Layers: {latent_data['layer_count']} | "
        f"Latents/layer: {latent_data['latent_dim']}"
    )

    return gif_path


# OPTIONAL: GENERATION + GIF
@torch.no_grad()
def generate_text(model, prompt_text, max_new_tokens=60, top_k=3, temperature=1.0):
    prompt = torch.tensor(
        bytearray(prompt_text, "utf-8"),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature,
    )

    decoded = bytes(
        output.to(torch.uint8).cpu().squeeze(0)
    ).decode(errors="backslashreplace")

    return decoded


def ask_and_animate(model, prompt_text, animate_prompt_only=True, max_new_tokens=60):
    """
    If animate_prompt_only=True:
        GIF is built from prompt tokens only.

    If False:
        generates continuation and builds GIF for full prompt+completion.
    """
    if animate_prompt_only:
        gif_path = create_prompt_gif(model, prompt_text)
        return {
            "prompt": prompt_text,
            "generated": None,
            "gif_path": gif_path,
        }

    full_text = generate_text(
        model=model,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        top_k=3,
        temperature=1.0,
    )

    gif_path = create_prompt_gif(model, full_text)
    return {
        "prompt": prompt_text,
        "generated": full_text,
        "gif_path": gif_path,
    }


# MAIN
def main():
    print(f"Loading model: {run_config.run}")

    model = load_model()

    tuning_model.seed_prompts(13)
    # keep your previous behaviour if you still want it
    # DATA_PATH = Path(__file__).parent / "inference" / f"{DATASET_NAME}.txt"
    # tuning_model.seed_prompts(13)
    # tuning_model.run_questions_from_file(run_config, device, DATA_PATH, model)
    # tuning_model.print_ranking_table(run_config, model, device)
    # tuning_model.print_greedy_table(run_config, model, device)

    prompt_text = input("\nEnter prompt for BDH latent animation: ").strip()
    if not prompt_text:
        prompt_text = "The capital of France is"

    result = ask_and_animate(
        model=model,
        prompt_text=prompt_text,
        # animate_prompt_only=True,   # change to False if you want prompt+generation
        animate_prompt_only=False,   # change to False if you want prompt+generation
        max_new_tokens=20,
    )

    print("\nDone.")
    print(f"GIF path: {result['gif_path']}")


if __name__ == "__main__":
    main()
