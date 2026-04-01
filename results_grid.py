import glob
import os
import re
import json
import numpy as np
import math
from factual_models import tuning_model
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
    # title = f"[{prefix_str}{min_num}-{prefix_str}{max_num}] Transformer runs"
    title = f"[{prefix_str}{min_num}-{prefix_str}{max_num}] Dragon hatchling runs"

# pio.templates.default = "plotly_dark"


# Metric colors (for grid only)
metric_colors = {
    "train": "rgb(80, 160, 255)",
    "val": "rgb(255, 120, 120)",
    "ppl": "rgb(120, 220, 120)",
}


def make_grid_figure():
    n_runs = len(runs)
    cols = 2
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
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    return fig


def make_single_figure():
    import plotly.express as px
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    marker_traces = []
    min_points = []  # <-- added

    for i, (name, data) in enumerate(runs.items()):
        steps = data["steps"]
        color = colors[i % len(colors)]

        # --- TRAIN ---
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data["train_loss"],
                mode="lines",
                name="Training loss",
                legendgroup="train",
                showlegend=(i == 0),
                line=dict(color=color, dash="dot"),
                hovertemplate=f"{name} Training loss<br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )

        # --- VAL ---
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data["val_loss"],
                mode="lines",
                name="Validation loss",
                legendgroup="val",
                showlegend=(i == 0),
                line=dict(color=color),
                hovertemplate=f"{name} Validation loss<br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )

        # --- PERPLEXITY ---
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=data["perplexity"],
                mode="lines",
                name="Perplexity",
                legendgroup="ppl",
                showlegend=(i == 0),
                # yaxis="y2",
                line=dict(color=color, dash="dash"),
                hovertemplate=f"{name} Perplexity<br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )

        # --- MIN VAL LOSS POINT ---
        val_losses = data["val_loss"]
        min_idx = int(np.argmin(val_losses))
        min_step = steps[min_idx]
        min_val = val_losses[min_idx]

        # marker
        # fig.add_trace(
        #     go.Scatter(
        #         x=[min_step],
        #         y=[min_val],
        #         mode="markers",
        #         # marker=dict(color=color, size=7),
        #         marker=dict(color="black", size=7),
        #         showlegend=False,
        #         hoverinfo="skip",
        #     )
        # )
        marker_traces.append(
            go.Scatter(
                x=[min_step],
                y=[min_val],
                mode="markers",
                # marker=dict(color=color, size=9, line=dict(
                marker=dict(color=color, size=7, line=dict(
                    width=1, color="black")),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        min_points.append((name, min_step, min_val, color))  # <-- added

        # annotation (arrow)
        # (moved below)

    # --- GROUP + ANNOTATE ---
    from collections import defaultdict
    groups = defaultdict(list)

    for name, x, y, color in min_points:
        # key = int(round(x / 500))
        key = x
        groups[key].append((name, x, y, color))

    y_min = min(min(data["val_loss"]) for data in runs.values())
    y_max = max(max(data["val_loss"]) for data in runs.values())

    sorted_groups = sorted(groups.values(), key=lambda g: g[0][1])

    prev_x = None
    stack_level = 0

    for group in sorted_groups:

        x = group[0][1]

        if prev_x is not None and abs(x - prev_x) < 1500:
            stack_level += 1
        else:
            stack_level = 0

        prev_x = x

        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=y_min,
            y1=y_max,
            line=dict(color="rgba(0,0,0,0.2)", width=2, dash="dash"),
            layer="below"
        )

        label_parts = []
        # for name, x_val, y_val, color in group:
        for name, x_val, y_val, color in sorted(group, key=lambda t: t[2], reverse=True):
            label_parts.append(
                f"<span style='color:{color}'>{name}: {y_val:.2f}</span>"
            )

        label_text = "<br>".join(label_parts)

        fig.add_annotation(
            x=x,
            y=min(y for _, _, y, _ in group),
            text=label_text,
            showarrow=False,
            yshift=40 + stack_level * 60,
            font=dict(size=13),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        )
    for trace in marker_traces:
        fig.add_trace(trace)

    fig.update_layout(
        # title=f"{title} combined plot",
        title=f"{title} combined plot",
        xaxis_title="Steps",
        # yaxis=dict(title="Loss / Perplexity"),
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="", overlaying="y", side="right"),
        legend=dict(orientation="h", y=.98, x=.98, xanchor="right"),
        uirevision="constant",
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,   #
        ticks="outside"
    )

    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside"
    )
    fig.update_layout(
        xaxis=dict(layer="above traces"),
        yaxis=dict(layer="above traces"),
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
                html.Button("Save as SVG", id="save-svg-btn"),
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
    Input("save-svg-btn", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def save_file(n_html, n_svg, fig_dict):
    ctx = dash.callback_context
    fig = go.Figure(fig_dict)
    fig.update_layout(
        width=1200,
        height=850,
    )
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "save-btn":
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

        # html_str = html_str.replace("<head>", f"<head>{custom_css}")
        return dict(content=html_str, filename="plot.html")

    if button_id == "save-svg-btn":
        from dash import dcc

        return dcc.send_bytes(
            lambda f: f.write(pio.to_image(
                fig, format="svg", engine="kaleido")),
            "plot.svg"
        )

    return dash.no_update


if __name__ == "__main__":
    app.run(debug=True)
