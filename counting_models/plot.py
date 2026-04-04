import math
from pathlib import Path
import dash
from dash import dcc, html
import plotly.graph_objs as go
import json


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


# logged data
# metrics = Path('counting_models') / "bdh_metrics.json"
metrics = Path("bdh_metrics.json")
bdh = load_metrics(metrics)

steps = bdh["steps"]

bdh_loss = bdh["val_loss"]

bdh_time = bdh["time_formatted"]

tf = load_metrics("transformer_metrics.json")
transformer_loss = tf["val_loss"]
# transformer_loss = []
transformer_time = tf["time_formatted"]
# transformer_loss = []

# helpers


def log_ticks_from_data(*series):
    all_vals = [v for s in series for v in s if v > 0]
    ymin, ymax = min(all_vals), max(all_vals)

    min_exp = math.floor(math.log10(ymin))
    max_exp = math.ceil(math.log10(ymax))

    return [10 ** e for e in range(min_exp, max_exp + 1)]


y_ticks = log_ticks_from_data(bdh_loss, transformer_loss)

# Find minimum points
bdh_min_idx = bdh_loss.index(min(bdh_loss))
tf_min_idx = transformer_loss.index(min(transformer_loss))

bdh_text = [""] * len(bdh_loss)
bdh_text[bdh_min_idx] = f"{bdh_loss[bdh_min_idx]:.4f}"

tf_text = [""] * len(transformer_loss)
tf_text[tf_min_idx] = f"{transformer_loss[tf_min_idx]:.4f}"


bdh_fig = go.Figure(
    data=[
        go.Scatter(
            x=steps,
            y=bdh_loss,
            mode="lines+markers+text",
            text=bdh_text,
            textposition="top center",
            name="BDH",
        )
    ]
)

bdh_fig.update_layout(
    title=dict(
        # text=f"(loss) Transformer compared to BDH (elapsed time: {transformer_time:.2f}s vs {bdh_time:.2f}s)",
        text=f"(val loss) Transformer vs BDH (time: {transformer_time} vs {bdh_time})",
        font=dict(size=18, family="Arial black"),
        x=0.5,
        y=0.92,
    ),
    margin=dict(t=60),
    xaxis=dict(
        title=dict(
            text="Training step",
            font=dict(size=14, family="Arial"),
        )
    ),
    yaxis=dict(
        title=dict(
            text="Loss",
            font=dict(size=14, family="Arial"),
        ),
        type="log",
        tickvals=y_ticks,
        tickformat=".2f",
    ),
)

transformer_fig = go.Figure(
    data=[
        go.Scatter(
            x=steps,
            y=transformer_loss,
            mode="lines+markers+text",
            text=tf_text,
            textposition="top center",
            name="Transformer",
        )
    ]
)

transformer_fig.update_layout(
    # margin=dict(t=40),
    xaxis=dict(
        title=dict(
            text="Training step",
            font=dict(size=14, family="Arial Black"),
        )
    ),
    yaxis=dict(
        title=dict(
            text="Loss",
            font=dict(size=14, family="Arial Black"),
        ),
        type="log",
        tickvals=y_ticks,
        tickformat=".2f",
    ),
)

# Overlay Transformer onto BDH

bdh_fig.add_trace(transformer_fig.data[0])

# Dash app

app = dash.Dash(__name__)

app.layout = html.Div(
    style={"padding": "20px"},
    children=[

        html.Div(
            style={
                "display": "flex",
                "alignItems": "flex-start",
                "gap": "20px",
            },
            children=[

                # LEFT: TEXT (30%)
                html.Div(
                    style={"width": "30%"},
                    children=[
                        html.H2([
                            "Counting task training loss comparison",
                            html.Br(),
                            "(dataset contains numbers in range [0, 999])",
                        ]),

                        html.Div([
                            "Hyperparameters: layers=6, embedding=256, heads=4; AdamW (lr=1e-3, wd=0.01),",
                            html.Br(),
                            " batch=64, steps=800 on identical data and hardware",
                        ]),
                        html.Br(),

                        html.Div("Both models solve correctly"),
                        html.Div("Input: 863"),
                        html.Div("Expected output: 864 865 866 867 868 869"),
                        html.Br(),

                        html.Div(
                            "Both models fail at extrapolating beyond training data"),
                        html.Br(),

                        html.Div("Input: 990"),
                        # html.Div(
                        #     "Expected output: 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003"
                        # ),
                        #
                        html.Div([
                            html.B("Expected output: "),
                            "990, 991, 992, 993, 994, 995, 996, 997, 998,",
                            html.Span(
                                "999",
                                style={
                                    "backgroundColor": "yellow",
                                    "padding": "2px 4px",
                                    "borderRadius": "4px",
                                    "fontWeight": "bold"
                                }
                            ),
                            ", 1000, 1001, 1002, 1003"
                        ]),

                        html.Div([
                            html.B("BDH output: "),
                            "990, 991, 992, 993, 994, 995, 996, 997, 998, ",
                            html.Span(
                                "999",
                                style={
                                    "backgroundColor": "yellow",
                                    "padding": "2px 4px",
                                    "borderRadius": "4px",
                                    "fontWeight": "bold"
                                }
                            ),
                            ", 900, 901, 902, 903"
                        ]),

                        html.Div([

                            html.B("Transformer output: "),
                            "990, 991, 992, 993, 994, 995, 996, 997, 998, ",
                            html.Span(
                                "999",
                                style={
                                    "backgroundColor": "yellow",
                                    "padding": "2px 4px",
                                    "borderRadius": "4px",
                                    "fontWeight": "bold"
                                }
                            ),
                            ", 840, 841, 842, 843"
                        ]),
                    ],
                ),

                # RIGHT: OVERLAID PLOT (70%)
                html.Div(
                    style={"width": "70%"},
                    children=[
                        dcc.Graph(figure=bdh_fig),
                    ],
                ),
            ],
        ),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
