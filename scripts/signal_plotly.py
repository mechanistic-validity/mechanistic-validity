"""Signal detection d' diagram using Plotly."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "public" / "figures" / "signal_variants"
OUT.mkdir(parents=True, exist_ok=True)

x = np.linspace(-3, 7, 500)

fig = make_subplots(rows=1, cols=2, subplot_titles=("Standard baselines", "High random baseline"),
                    horizontal_spacing=0.08)

panels = [
    {"mu_signal": 3.0, "dprime": "d′ = 2.0", "col": 1},
    {"mu_signal": 1.8, "dprime": "d′ = 0.8", "col": 2},
]
mu_noise = 1.0

for p in panels:
    noise = norm.pdf(x, mu_noise, 1.0)
    signal = norm.pdf(x, p["mu_signal"], 1.0)
    col = p["col"]

    fig.add_trace(go.Scatter(x=x, y=noise, fill='tozeroy', fillcolor='rgba(239,68,68,0.25)',
                             line=dict(color='#ef4444', width=2), name='Non-circuit',
                             showlegend=(col == 1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=x, y=signal, fill='tozeroy', fillcolor='rgba(59,130,246,0.25)',
                             line=dict(color='#3b82f6', width=2), name='Circuit',
                             showlegend=(col == 1)), row=1, col=col)

    criterion = (mu_noise + p["mu_signal"]) / 2
    fig.add_vline(x=criterion, line_dash="dash", line_color="#555", line_width=1.5,
                  annotation_text="criterion", annotation_position="top", row=1, col=col)

    fig.add_annotation(x=(mu_noise + p["mu_signal"]) / 2, y=0.06,
                       text=f"<b>{p['dprime']}</b>", showarrow=False,
                       font=dict(size=14, color="#22a352"), row=1, col=col)
    fig.add_annotation(x=mu_noise, y=0.04, ax=p["mu_signal"], ay=0.04,
                       showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
                       arrowcolor="#22a352", axref=f"x{col}", ayref=f"y{col}", row=1, col=col)
    fig.add_annotation(x=p["mu_signal"], y=0.04, ax=mu_noise, ay=0.04,
                       showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
                       arrowcolor="#22a352", axref=f"x{col}", ayref=f"y{col}", row=1, col=col)

fig.update_layout(
    title=dict(text="Signal Detection Framework for Circuit Identification",
               font=dict(size=20, color="#333"), x=0.5),
    width=1000, height=450,
    plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
    font=dict(family="Arial", color="#333"),
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=80, b=60),
)

for i in [1, 2]:
    fig.update_xaxes(title_text="Instrument score", range=[-2.5, 6.5], row=1, col=i,
                     gridcolor="#eee", zeroline=False)
    fig.update_yaxes(title_text="Density" if i == 1 else "", range=[0, 0.48], row=1, col=i,
                     gridcolor="#eee", zeroline=False)

fig.write_html(str(OUT / "signal_plotly.html"), include_plotlyjs="cdn")
fig.write_image(str(OUT / "signal_plotly.png"), scale=2)
print(f"Written to {OUT}")
