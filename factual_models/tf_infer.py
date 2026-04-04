import torch
from template.gpu_support import GPUSupport
from pathlib import Path
from factual_models import tf_model
from factual_models import tuning_model

print("Available model files:")
model_dir = Path(__file__).parent / "models"
for path in sorted(model_dir.iterdir()):
    if path.is_file():
        print(path.name)

# select run config
run_config = tuning_model.interact('transformer')
# run_config = getattr(tuning_model, "D10")
DATASET_NAME = tuning_model.datasets[run_config.run[0]]

#     os.path.join(os.path.dirname(__file__), f"inference/{run_config.run}")
MODEL_PATH = Path(__file__).parent / "models" / f"{run_config.run}.pt"

print(f"\033[42m\033[30m{DATASET_NAME=}\033[0m")
print(f"\033[43m\033[30m{run_config}\033[0m")

GPUSupport()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = tf_model.Transformer(
        n_layer=run_config.tf_n_layer,
        d_model=run_config.tf_d_model,
        dropout=run_config.tf_dropout,
        n_head=run_config.tf_n_head,
        mlp_mult=run_config.tf_mlp_mult,
        vocab_size=run_config.tf_vocab_size,
    ).to(device)

    # checkpoint = torch.load(MODEL_PATH, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])

    state_dict = torch.load(MODEL_PATH, map_location=device)[
        "model_state_dict"]

    # fix: remove "_orig_mod." prefix
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

    tuning_model.run_questions_from_file(run_config, device, DATA_PATH, model)

    # print("\nGENERATED TEXT \n")
    # print(result)
    # print_markdown_table(prompt, result)
