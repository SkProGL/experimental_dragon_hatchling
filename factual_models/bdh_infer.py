import torch
from template.gpu_support import GPUSupport
from pathlib import Path
from factual_models import bdh_model as bdh
from factual_models import tuning_model

print("Available model files:")
model_dir = Path(__file__).parent / "models"
for path in sorted(model_dir.iterdir()):
    if path.is_file():
        print(path.name)

# select run config
run_config = tuning_model.interact()
# run_config = getattr(tuning_model, "A10")
# run_config = getattr(tuning_model, "C2")

DATASET_NAME = tuning_model.datasets[run_config.run[0]]
#     os.path.join(os.path.dirname(__file__), f"inference/{run_config.run}")
MODEL_PATH = Path(__file__).parent / "models" / f"{run_config.run}.pt"
print(f"\033[42m\033[30m{DATASET_NAME=}\033[0m")
print(f"\033[43m\033[30m{run_config}\033[0m")

GPUSupport()
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

    # checkpoint = torch.load(MODEL_PATH, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])

    state_dict = torch.load(MODEL_PATH, map_location=device)[
        "model_state_dict"]

    # fix: removed "_orig_mod." prefix
    new_state_dict = {k.replace("_orig_mod.", ""):
                      v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.eval()
    return model


def print_markdown_table(prompt, output):
    # escape pipes/newlines for markdown safety
    safe_prompt = prompt.replace("|", "\\|").replace("\n", "<br>")
    safe_output = output.replace("|", "\\|").replace("\n", "<br>")

    print("\n| Prompt | Output |")
    print("|--------|--------|")
    print(f"| {safe_prompt} | {safe_output} |")


def main():
    print(f"Loading model: {run_config.run}")

    DATA_PATH = Path(__file__).parent / "inference" / f"{DATASET_NAME}.txt"
    model = load_model()

    # prompt = "Gravity\n\nGravity is"
    # result = generate_text(device, model, prompt)

    tuning_model.run_questions_from_file(run_config, device, DATA_PATH, model)

    # print("\n=== GENERATED TEXT ===\n")
    # print(result)
    # print_markdown_table(prompt, result)
