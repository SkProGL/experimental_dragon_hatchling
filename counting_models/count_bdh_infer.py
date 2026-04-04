import torch
from pathlib import Path
import os
from counting_models import count_bdh_model

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# load model

CKPT_PATH = Path('counting_models') / "A2_rule_bdh.pt"

assert os.path.exists(CKPT_PATH), "Model checkpoint not found"

checkpoint = torch.load(CKPT_PATH, map_location=device)

# rebuild config
config = count_bdh_model.BDHConfig(**checkpoint["config"])

# load model
model = count_bdh_model.BDH(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded successfully\n")

# helper


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

    decoded = bytes(
        result.to(torch.uint8).cpu().squeeze(0)
    ).decode(errors="backslashreplace")

    return decoded


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
