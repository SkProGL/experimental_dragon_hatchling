import torch
from pathlib import Path

from factual_models import bdh_model as bdh
from factual_models import tuning_model

# select run config
run_config = tuning_model.A4  # 👈 use A4 directly

MODEL_PATH = Path(f"results/{run_config.run}.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# build model config from run config
def build_bdh_config(run_config):
    return bdh.BDHConfig(
        n_layer=run_config.bdh_n_layer,
        n_embd=run_config.bdh_n_embd,
        dropout=run_config.bdh_dropout,
        n_head=run_config.bdh_n_head,
        mlp_internal_dim_multiplier=run_config.bdh_mlp_internal_dim_multiplier,
        vocab_size=run_config.bdh_vocab_size,
    )


def load_model():
    config = build_bdh_config(run_config)

    model = bdh.BDH(config).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model


# generation
def generate_text(model, prompt_text, max_new_tokens=100, top_k=3, temperature=1.0):
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
            temperature=temperature,
        )

    decoded = bytes(
        output.to(torch.uint8).cpu().squeeze(0)
    ).decode(errors="backslashreplace")

    return decoded


def print_markdown_table(prompt, output):
    # escape pipes/newlines for markdown safety
    safe_prompt = prompt.replace("|", "\\|").replace("\n", "<br>")
    safe_output = output.replace("|", "\\|").replace("\n", "<br>")

    print("\n| Prompt | Output |")
    print("|--------|--------|")
    print(f"| {safe_prompt} | {safe_output} |")


if __name__ == "__main__":
    print(f"Loading model: {run_config.run}")

    model = load_model()

    prompt = "Gravity\n\nGravity is"
    result = generate_text(model, prompt)

    # print("\n=== GENERATED TEXT ===\n")
    # print(result)
    print_markdown_table(prompt, result)
