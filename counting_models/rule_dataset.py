def generate_rule_dataset(
    num_lines: int = 7000,
    min_start: int = 1,
    max_start: int = 20,
    min_steps: int = 2,
    max_steps: int = 6,
    seed: int = 0,
) -> str:
    """
    Generate a rule-based reasoning dataset.

    Each sample defines:
    - starting number
    - rule (even/odd transformation)
    - number of steps
    - full step-by-step sequence

    Designed to test multi-step reasoning and extrapolation.
    """

    import random
    random.seed(seed)

    # Define possible rules
    rules = [
        ("even:+3,odd:*2", lambda x: x + 3 if x % 2 == 0 else x * 2),
        ("even:+2,odd:+1", lambda x: x + 2 if x % 2 == 0 else x + 1),
        ("even:*2,odd:+3", lambda x: x * 2 if x % 2 == 0 else x + 3),
        ("even:+1,odd:*2", lambda x: x + 1 if x % 2 == 0 else x * 2),
    ]

    lines = []

    for _ in range(num_lines):
        start = random.randint(min_start, max_start)
        steps = random.randint(min_steps, max_steps)

        rule_name, rule_fn = random.choice(rules)

        seq = [start]
        current = start

        for _ in range(steps):
            current = rule_fn(current)
            seq.append(current)

        seq_str = " -> ".join(str(x) for x in seq)

        line = f"start={start}; rule={rule_name}; steps={steps};\n{seq_str}"
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    data = generate_rule_dataset()
    filename = "rule_data.txt"

    print(f"\033[43m\033[30m{filename}\033[0m")

    with open(filename, "w") as f:
        f.write(data)
