import json
import numpy as np
import plotly.graph_objects as go
import glob
import os

# load JSON
name = "A1"

names = sorted(
    os.path.basename(f)[:-5]
    for f in glob.glob("results/[A-Z]*.json")
)

print("| Run | Layers | Emb | Heads | MLP Mult | LR | Batch | Weight Decay | Iterations |")
print("| --- | ------ | --- | ----- | -------- | -- | ----- | ------------ | ---------- |")
for name in names:
    with open(f"results/{name}.json", "r") as f:
        data = json.load(f)['config']
        # data = json.load(f)
        print(f"| {data['run']} "
              f"| {data.get('bdh_n_layer', data.get('tf_n_layer'))} "
              f"| {data.get('bdh_n_embd', data.get('tf_d_model'))} "
              f"| {data.get('bdh_n_head', data.get('tf_n_head'))} "
              f"| {data.get('bdh_mlp_internal_dim_multiplier', data.get('tf_mlp_mult'))} "
              f"| {data['train_learning_rate']:.0e} "
              f"| {data['train_batch_size']} "
              f"| {data['train_weight_decay']} "
              f"| {int(data['train_max_iters']/1000)}k |")

print()
print("| Run | Train Loss | Val Loss | Perplexity | Sparsity | Latent/Layer | Time (hrs) |")
print("| --- | ---------- | -------- | ---------- | -------- | ------------ | ---------- |")

for name in names:
    with open(f"results/{name}.json", "r") as f:
        data = json.load(f)['metrics']
        # data = json.load(f)

        steps = data["steps"]
        train_loss = data["train_loss"]
        val_loss = data["val_loss"]
        perplexity = data["perplexity"]

        best_i = np.argmin(val_loss)

        train_rep = train_loss[best_i]
        val_rep = val_loss[best_i]
        ppl_rep = np.exp(val_rep)
        if int(data['elapsed_time']//60) > 0:
            time_str = f"{int(data['elapsed_time']//60)}h {int(data['elapsed_time'] % 60)}m"
        else:
            time_str = f"{int(data['elapsed_time'] % 60)}m"
        print(f"| {name} "
              f"| {train_rep:.2f} "
              f"| {val_rep:.2f} "
              f"| {ppl_rep:.2f} "
              f'| {("{:.3f}".format(data["sparsity_ratio"]) if data["sparsity_ratio"] is not None else "-")} '
              f'| {(data["total_latent_per_layer"] if data["total_latent_per_layer"] is not None else "-")} '
              f"| {time_str} |")
        # print(data)

# import json
# import numpy as np
# import plotly.graph_objects as go
# import glob
# import os
#
# # load JSON
# name = "A1"
#
# names = sorted(
#     os.path.basename(f)[:-5]
#     for f in glob.glob("results/[A-Z]*.json")
# )
#
# print("| Run | Layers | Emb | Heads | MLP Mult | LR | Batch | Weight Decay | Iterations |")
# print("| --- | ------ | --- | ----- | -------- | -- | ----- | ------------ | ---------- |")
# for name in names:
#     with open(f"results/{name}.json", "r") as f:
#         data = json.load(f)['config']
#         # data = json.load(f)
#         print(f"| {data['run']} "
#               f"| {data['bdh_n_layer']} "
#               f"| {data['bdh_n_embd']} "
#               f"| {data['bdh_n_head']} "
#               f"| {data['bdh_mlp_internal_dim_multiplier']} "
#               f"| {data['train_learning_rate']:.0e} "
#               f"| {data['train_batch_size']} "
#               f"| {data['train_weight_decay']} "
#               f"| {int(data['train_max_iters']/1000)}k |")
#
# print()
# print("| Run | Train Loss | Val Loss | Perplexity | Sparsity | Latent/Layer | Time (hrs) |")
# print("| --- | ---------- | -------- | ---------- | -------- | ------------ | ---------- |")
#
# for name in names:
#     with open(f"results/{name}.json", "r") as f:
#         data = json.load(f)['metrics']
#         # data = json.load(f)
#
#         steps = data["steps"]
#         train_loss = data["train_loss"]
#         val_loss = data["val_loss"]
#         perplexity = data["perplexity"]
#
#         best_i = np.argmin(val_loss)
#
#         train_rep = train_loss[best_i]
#         val_rep = val_loss[best_i]
#         ppl_rep = np.exp(val_rep)
#         if int(data['elapsed_time']//60) > 0:
#             time_str = f"{int(data['elapsed_time']//60)}h {int(data['elapsed_time'] % 60)}m"
#         else:
#             time_str = f"{int(data['elapsed_time'] % 60)}m"
#         print(f"| {name} "
#               f"| {train_rep:.2f} "
#               f"| {val_rep:.2f} "
#               f"| {ppl_rep:.2f} "
#               f'| {data["sparsity_ratio"]:.3f} '
#               f'| {data["total_latent_per_layer"]} '
#               f"| {time_str} |")
#         # print(data)
#
