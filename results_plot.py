import json
import numpy as np
import plotly.graph_objects as go

# load JSON
name = "A1"

print("| Run | Layers | Emb | Heads | MLP Mult | LR | Batch | Weight Decay | Iterations |")
print("| --- | ------ | --- | ----- | -------- | -- | ----- | ------------ | ---------- |")
for i in range(1, 11):
    name = f"A{i}"
    with open(f"results/{name}.json", "r") as f:
        data = json.load(f)['config']
        # data = json.load(f)
        print(f"| {data['run']} "
              f"| {data['bdh_n_layer']} "
              f"| {data['bdh_n_embd']} "
              f"| {data['bdh_n_head']} "
              f"| {data['bdh_mlp_internal_dim_multiplier']} "
              f"| {data['train_learning_rate']:.0e} "
              f"| {data['train_batch_size']} "
              f"| {data['train_weight_decay']} "
              f"| {int(data['train_max_iters']/1000)}k |")

print()
print("| Run | Train Loss | Val Loss | Perplexity | Sparsity | Latent/Layer | Time (hrs) |")
print("| --- | ---------- | -------- | ---------- | -------- | ------------ | ---------- |")

for i in range(1, 11):
    name = f"A{i}"
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
              f'| {data["sparsity_ratio"]:.3f} '
              f'| {data["total_latent_per_layer"]} '
              f"| {time_str} |")
        # print(data)


title = f"{name} Metrics | "
f"Avg Training (loss): {train_rep:.4f} | "
f"Avg Validation (loss): {val_rep:.4f} | "
f"Avg perplexity: {ppl_rep:.4f}",
# create figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=steps, y=train_loss,
    mode='lines',
    name='Train Loss'
))

fig.add_trace(go.Scatter(
    x=steps, y=val_loss,
    mode='lines',
    name='Val Loss'
))

fig.add_trace(go.Scatter(
    x=steps, y=perplexity,
    mode='lines',
    name='Perplexity',
    yaxis='y2'  # optional secondary axis
))

# Layout with secondary y-axis
fig.update_layout(
    title=f"{name} Metrics | "
    f"Avg Training (loss): {train_rep:.4f} | "
    f"Avg Validation (loss): {val_rep:.4f} | "
    f"Avg perplexity: {ppl_rep:.4f}",
    xaxis_title="Steps",
    yaxis=dict(title="Loss"),
    yaxis2=dict(
        title="Perplexity",
        overlaying='y',
        side='right'
    ),
    template="plotly_white"
)

fig.show()
