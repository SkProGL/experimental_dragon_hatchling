import re
from collections import defaultdict
import matplotlib.pyplot as plt
from pycirclize import Circos
import pandas as pd
import numpy as np
import string
import json

# --- synthetic tree generator (unchanged) ---


def load_data(folder="results"):
    import glob
    import os
    import json
    import numpy as np

    names = sorted(
        os.path.basename(f)[:-5]
        for f in glob.glob(os.path.join(folder, "[A-Z]*.json"))
    )

    result = {}

    for name in names:
        with open(os.path.join(folder, f"{name}.json"), "r") as f:
            data = json.load(f)["metrics"]

        steps = data["steps"]
        train_loss = data["train_loss"]
        val_loss = data["val_loss"]

        best_i = int(np.argmin(val_loss))

        train_rep = float(train_loss[best_i])
        val_rep = float(val_loss[best_i])
        ppl_rep = float(np.exp(val_rep))

        # time formatting (same as your code)
        time_min = int(round(data["elapsed_time"]))
        # if int(data["elapsed_time"] // 60) > 0:
        #     time_str = f"{int(data['elapsed_time']//60)}h {int(data['elapsed_time'] % 60)}m"
        # else:
        #     time_str = f"{int(data['elapsed_time'] % 60)}m"

        result[name] = {
            "training_loss": round(train_rep, 2),
            "validation_loss": round(val_rep, 2),
            "perplexity": round(ppl_rep, 2),
            "sparsity": (
                round(data["sparsity_ratio"], 3)
                if data["sparsity_ratio"] is not None else None
            ),
            "latent_layer": (
                data["total_latent_per_layer"]
                if data["total_latent_per_layer"] is not None else None
            ),
            "time": f"{time_min}m",
        }

    # print(json.dumps(result, indent=2))
    return result


def build_balanced_tree(labels):
    nodes = [f"{label}:1" for label in labels]
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                new_nodes.append(f"({nodes[i]},{nodes[i+1]}):1")
            else:
                new_nodes.append(nodes[i])
        nodes = new_nodes
    return nodes[0]


def generate_tree_from_run_names(run_names):
    pattern = re.compile(r"^([A-Z]+)(\d+)$")
    grouped = defaultdict(list)

    for name in run_names:
        m = pattern.match(name)
        if not m:
            continue
        prefix, num = m.groups()
        grouped[prefix].append((int(num), name))

    group_trees = []
    for prefix in sorted(grouped):
        labels = [name for _, name in sorted(
            grouped[prefix], key=lambda x: x[0])]
        group_trees.append(build_balanced_tree(labels))

    return build_balanced_tree(group_trees) + ";"

# def build_balanced_tree(labels):
#     nodes = [f"{l}:1" for l in labels]
#     while len(nodes) > 1:
#         new_nodes = []
#         for i in range(0, len(nodes), 2):
#             if i + 1 < len(nodes):
#                 new_nodes.append(f"({nodes[i]},{nodes[i+1]}):1")
#             else:
#                 new_nodes.append(nodes[i])
#         nodes = new_nodes
#     return nodes[0]
#
#
# def generate_grouped_tree(groups, per_group):
#     group_trees = []
#     for g in groups:
#         labels = [f"{g}{i}" for i in range(1, per_group + 1)]
#         group_tree = build_balanced_tree(labels)
#         group_trees.append(group_tree)
#     return build_balanced_tree(group_trees) + ";"
# tree_file = generate_grouped_tree(
#     groups=list(string.ascii_uppercase[:4]),
#     per_group=10
# )

# --- HARD-CODED JSON DATA (extended) ---
# json_data = {
#     "A1": {"training_loss": 10, "validation_loss": 20, "perplexity": 30, "sparsity": 0.842, "latent_layer": 16384, "time": "1h 288m"},
#     "A2": {"training_loss": 15, "validation_loss": 25, "perplexity": 35, "sparsity": 0.864, "latent_layer": 24576, "time": "1h 288m"},
#     "A3": {"training_loss": 20, "validation_loss": 30, "perplexity": 40, "sparsity": 0.874, "latent_layer": 32768, "time": "1h 288m"},
#     "A4": {"training_loss": 25, "validation_loss": 35, "perplexity": 45, "sparsity": 0.863, "latent_layer": 49152, "time": "1h 288m"},
#     "A5": {"training_loss": 30, "validation_loss": 40, "perplexity": 50, "sparsity": 0.876, "latent_layer": 49152, "time": "1h 288m"},
#     "A6": {"training_loss": 35, "validation_loss": 45, "perplexity": 55, "sparsity": 0.879, "latent_layer": 65536, "time": "1h 288m"},
#     "A7": {"training_loss": 40, "validation_loss": 50, "perplexity": 60, "sparsity": 0.864, "latent_layer": 65536, "time": "1h 288m"},
#     "A8": {"training_loss": 45, "validation_loss": 55, "perplexity": 65, "sparsity": 0.891, "latent_layer": 131072, "time": "1h 288m"},
#     "A9": {"training_loss": 50, "validation_loss": 60, "perplexity": 70, "sparsity": 0.884, "latent_layer": 131072, "time": "1h 288m"},
#     "A10": {"training_loss": 55, "validation_loss": 65, "perplexity": 75, "sparsity": 0.869, "latent_layer": 24576, "time": "1h 288m"},
# }
#
# # auto-fill B/C/D with same pattern (keeps your structure intact)
# for prefix in ["B", "C", "D"]:
#     for i in range(1, 11):
#         json_data[f"{prefix}{i}"] = json_data[f"A{i}"]


json_data = load_data()
tree_file = generate_tree_from_run_names(json_data.keys())
circos, tv = Circos.initialize_from_tree(
    tree_file,
    format="newick",
    start=10,
    end=350,
    ignore_branch_length=True,
    # leaf_label_rmargin=-12,
    line_kws=dict(color="none", lw=0),
)

# label_formatter=lambda t: f"• {t}" if t.endswith("0") else t,
# --- convert JSON → DataFrame ---
df = pd.DataFrame.from_dict(json_data, orient="index")
df = df.loc[tv.leaf_labels]
df["sparsity"] = pd.to_numeric(df["sparsity"], errors="coerce")
df["latent_layer"] = pd.to_numeric(df["latent_layer"], errors="coerce")
df["sparsity"] = df["sparsity"].fillna(0.0)
df["latent_layer"] = df["latent_layer"].fillna(0.0)

print(df.head())

sector = tv.track.parent_sector

# --- NEW TRACKS ---

track1 = sector.add_track((45, 55))
track1.heatmap(df["training_loss"].to_numpy(), cmap="Reds_r",
               show_value=True, rect_kws=dict(ec="grey", lw=1.5))

track2 = sector.add_track((55, 65))
track2.heatmap(df["validation_loss"].to_numpy(), cmap="Blues_r",
               show_value=True, rect_kws=dict(ec="grey", lw=1.5))


track3 = sector.add_track((65, 75))
track3.heatmap(df["perplexity"].to_numpy(), cmap="Greens_r",
               show_value=True, rect_kws=dict(ec="grey", lw=1.5))


track4 = sector.add_track((75, 85))
track4.heatmap(df["sparsity"].to_numpy(), cmap="Purples",
               show_value=True, rect_kws=dict(ec="grey", lw=1.5))

# track5 = sector.add_track((60, 65))
# track5.heatmap(df["latent_layer"].to_numpy(), cmap="Oranges",
#                show_value=True, rect_kws=dict(ec="grey", lw=0.5))
time_track5 = sector.add_track((85, 90))
time_track5.axis()  # ← THIS FIXES IT
x = np.arange(0, tv.leaf_num) + 0.5
labels = df["time"].to_list()
for xi, label in zip(x, labels):
    time_track5.text(label, x=xi, size=12)

bar_track6 = sector.add_track((90, 100), r_pad_ratio=0.1)
bar_track6.axis()
bar_track6.grid()
x = np.arange(0, tv.leaf_num) + 0.5
y = df["latent_layer"].to_numpy()
bar_track6.bar(x, y, width=0.6, color="orange")


dot_track = sector.add_track((43, 43))  # inner circle zone

x = np.arange(0, tv.leaf_num) + 0.5
# mark only A10, B10, etc.
mask = [label.endswith("0") for label in tv.leaf_labels]
dot_track.scatter(
    x=np.array(x)[mask],
    y=[1] * sum(mask),  # constant radius
    s=20,
    color="grey",
)

circos.text("Training loss", r=track1.r_center, color="red")
circos.text("Validation loss", r=track2.r_center, color="blue")
circos.text("Perplexity", r=track3.r_center, color="green")
circos.text("Sparsity", r=track4.r_center, color="purple")
# circos.text("Latent", r=track5.r_center, color="orange")
circos.text("Time (minutes)", r=time_track5.r_center, color="black")
circos.text("Latent size", r=bar_track6.r_center, color="orange")
fig = circos.plotfig()
circos.ax.text(
    0, 0, "Comparison of BDH and Transformer runs\n(A–F, 1–10)", ha="center", va="center", fontsize=14)

circos.text("Time", r=time_track5.r_center, color="black")
plt.show()
