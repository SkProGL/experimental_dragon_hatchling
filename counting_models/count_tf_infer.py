import os
from pathlib import Path

import torch

from counting_models.count_tf_train import TransformerModel

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# load model
CKPT_PATH = Path("counting_models") / "A2_rule_transformer.pt"

assert os.path.exists(CKPT_PATH), "Model checkpoint not found"

checkpoint = torch.load(CKPT_PATH, map_location=device)

model = TransformerModel().to(device)

state_dict = checkpoint["model_state_dict"]

# compatibility patch for older compiled checkpoints
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("_orig_mod.", "")
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

print("Transformer loaded\n")


def generate(prompt_text, max_new_tokens=50):
    prompt = torch.tensor(
        bytearray(prompt_text, "utf-8"),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    result = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        top_k=3,
        # temperature=1.0,
    )

    return bytes(result.squeeze(0).cpu().numpy()).decode(errors="replace")


def main():
    tests = [
        ("interpolation", "start=3; rule=even:+3,odd:*2; steps=3;\n"),
        ("interpolation", "start=5; rule=even:+2,odd:+1; steps=4;\n"),
        ("extrapolation", "start=3; rule=even:+3,odd:*2; steps=8;\n"),
        ("extrapolation", "start=4; rule=even:*2,odd:+3; steps=10;\n"),

        ("extrapolation", "start=7; rule=even:+3,odd:*2; steps=7;\n"),
        ("extrapolation", "start=10; rule=even:+6,odd:+5; steps=1;\n"),
    ]

    for label, t in tests:
        # print(f"prompt ({label}):")
        # print(t)
        out = generate(t)
        print(f"OUTPUT ({label}):")
        print(out)
        print("=" * 60)
