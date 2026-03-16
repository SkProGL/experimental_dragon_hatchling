import json
import numpy as np
import plotly.graph_objects as go

# Load JSON
# with open("A2_metrics.json", "r") as f:
name = "A3"
with open(f"{name}_metrics.json", "r") as f:
    data = json.load(f)

steps = data["steps"]
train_loss = data["train_loss"]
val_loss = data["val_loss"]
perplexity = data["perplexity"]

# Calculate averages
avg_train = np.mean(train_loss)
avg_val = np.mean(val_loss)
avg_ppl = np.mean(perplexity)

# Create figure
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
    f"Avg Training (loss): {avg_train:.4f} | "
    f"Avg Validation (loss): {avg_val:.4f} | "
    f"Avg perplexity: {avg_ppl:.4f}",
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
