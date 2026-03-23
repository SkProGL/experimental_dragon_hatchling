import glob
import os
import re
import json
import numpy as np
import math

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
runs = {}

files = glob.glob("results/*.json")

pattern = re.compile(r"([A-Z]+)(\d+)\.json$")

parsed = []

for filepath in files:
    filename = os.path.basename(filepath)
    match = pattern.match(filename)

    if not match:
        continue

    prefix, num = match.groups()
    num = int(num)

    parsed.append((prefix, num, filepath))

# sort like A1, A2, ..., B1, B2
parsed.sort(key=lambda x: (x[0], x[1]))

for prefix, num, filepath in parsed:
    name = f"{prefix}{num}"

    with open(filepath, "r") as f:
        data = json.load(f)["metrics"]

    runs[name] = {
        "steps": data["steps"],
        "train_loss": data["train_loss"],
        "val_loss": data["val_loss"],
        "perplexity": data["perplexity"],
    }
if parsed:
    prefixes = sorted(set(p for p, _, _ in parsed))
    min_num = min(n for _, n, _ in parsed)
    max_num = max(n for _, n, _ in parsed)

    prefix_str = ",".join(prefixes)
    # title = f"[{prefix_str}{min_num}-{prefix_str}{max_num}] BDH TinyStories dataset runs"
    title = f"[{prefix_str}{min_num}-{prefix_str}{max_num}] BDH En-Wiki dataset runs"

pio.templates.default = "plotly_dark"


# Metric colors (for grid only)
metric_colors = {
    "train": "rgb(80, 160, 255)",
    "val": "rgb(255, 120, 120)",
    "ppl": "rgb(120, 220, 120)",
}


def make_grid_figure():
    n_runs = len(runs)
    cols = 3
    rows = math.ceil(n_runs / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(runs.keys()),
        specs=[[{"secondary_y": True}
                for _ in range(cols)] for _ in range(rows)],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    row = 1
    col = 1

    for name, data in runs.items():
        steps = data["steps"]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data["train_loss"],
                mode="lines",
                name="Training loss",
                legendgroup="train",
                line=dict(color=metric_colors["train"]),
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data["val_loss"],
                mode="lines",
                name="Validation loss",
                legendgroup="val",
                line=dict(color=metric_colors["val"]),
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data["perplexity"],
                mode="lines",
                name="Perplexity",
                legendgroup="ppl",
                line=dict(color=metric_colors["ppl"]),
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
            secondary_y=True,
        )

        col += 1
        if col > cols:
            col = 1
            row += 1

    fig.update_layout(
        title=f"{title} grid",
        height=320 * rows,
        autosize=True,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        uirevision="constant",
    )

    return fig


def make_single_figure():
    fig = go.Figure()

    for i, (name, data) in enumerate(runs.items()):
        steps = data["steps"]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data["train_loss"],
                mode="lines",
                name="Training loss",
                legendgroup="train",
                showlegend=(i == 0),
                hovertemplate=f"{name} Training loss<br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data["val_loss"],
                mode="lines",
                name="Validation loss",
                legendgroup="val",
                showlegend=(i == 0),
                hovertemplate=f"{name} Validation loss<br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data["perplexity"],
                mode="lines",
                name="Perplexity",
                legendgroup="ppl",
                showlegend=(i == 0),
                yaxis="y2",
                hovertemplate=f"{name} Perplexity<br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{title} combined plot",
        xaxis_title="Steps",
        yaxis=dict(title="Loss / Perplexity"),
        yaxis2=dict(title="Perplexity", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        uirevision="constant",
    )

    return fig


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.RadioItems(
                    id="view-toggle",
                    options=[
                        {"label": "Grid", "value": "grid"},
                        {"label": "Single", "value": "single"},
                    ],
                    value="single",
                    inline=True,
                ),
                html.Button("Save as HTML", id="save-btn"),
                dcc.Download(id="download"),
            ],
            style={"padding": "10px", "display": "flex", "gap": "10px"},
        ),
        dcc.Graph(
            id="graph",
            style={"height": "80vh", "width": "85vw", "margin": "auto"},
            config={"responsive": True},
        ),
    ],
    style={"width": "100%", "height": "100%", "margin": "0"},
)


@app.callback(
    Output("graph", "figure"),
    Input("view-toggle", "value"),
)
def update_view(mode):
    if mode == "grid":
        return make_grid_figure()
    return make_single_figure()


@app.callback(
    Output("download", "data"),
    Input("save-btn", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def save_html(n_clicks, fig_dict):
    fig = go.Figure(fig_dict)

    html_str = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")

    custom_css = """
        <style>
            body {
                background-color: black;
                color: white;
                margin: 0;
            }
        </style>
        """

    html_str = html_str.replace("<head>", f"<head>{custom_css}")

    return dict(content=html_str, filename="plot.html")


if __name__ == "__main__":
    app.run(debug=True)
