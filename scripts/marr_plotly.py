"""Marr's levels diagram using Plotly shapes + annotations."""

import plotly.graph_objects as go
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "public" / "figures" / "marr_variants"
OUT.mkdir(parents=True, exist_ok=True)

fig = go.Figure()
fig.update_layout(
    width=900, height=650,
    plot_bgcolor="#fafafa",
    paper_bgcolor="#fafafa",
    xaxis=dict(range=[0, 10], visible=False, fixedrange=True),
    yaxis=dict(range=[0, 10], visible=False, fixedrange=True),
    margin=dict(l=20, r=20, t=60, b=20),
    title=dict(
        text="Description Levels and Evidence Requirements",
        font=dict(size=22, color="#1a1a2e", family="Arial"),
        x=0.5,
    ),
    font=dict(family="Arial"),
)

colors = {
    "comp": "#3b82f6", "comp_bg": "rgba(219,234,254,0.7)",
    "algo": "#2a9d8f", "algo_bg": "rgba(209,250,229,0.7)",
    "impl": "#d97706", "impl_bg": "rgba(254,243,199,0.7)",
    "sub_bg": "rgba(255,247,237,0.8)",
    "drift": "#e63946",
}

# Implementational (large box)
fig.add_shape(type="rect", x0=1.2, y0=0.8, x1=8.8, y1=4.2,
              fillcolor=colors["impl_bg"], line=dict(color=colors["impl"], width=2),
              layer="below")

# Sub-mode boxes
subs = [("Topographic", 1.6, 3.4), ("Connectomic", 3.3, 5.1),
        ("Activation-<br>statistical", 5.0, 6.8), ("Functional", 6.7, 8.5)]
for label, x0, x1 in subs:
    fig.add_shape(type="rect", x0=x0, y0=1.0, x1=x1, y1=2.2,
                  fillcolor=colors["sub_bg"], line=dict(color=colors["impl"], width=1.2),
                  layer="below")
    fig.add_annotation(x=(x0+x1)/2, y=1.6, text=f"<b>{label}</b>",
                       showarrow=False, font=dict(size=12, color="#78350f"))

fig.add_annotation(x=5, y=3.6, text="<b>Implementational</b>",
                   showarrow=False, font=dict(size=20, color="#92400e"))

# Algorithmic
fig.add_shape(type="rect", x0=2.5, y0=5.0, x1=7.5, y1=6.4,
              fillcolor=colors["algo_bg"], line=dict(color=colors["algo"], width=2),
              layer="below")
fig.add_annotation(x=5, y=5.9, text="<b>Algorithmic</b>",
                   showarrow=False, font=dict(size=20, color="#115e59"))
fig.add_annotation(x=5, y=5.3, text="What procedure is executed",
                   showarrow=False, font=dict(size=12, color="#1a4a42"))

# Computational
fig.add_shape(type="rect", x0=2.5, y0=7.2, x1=7.5, y1=8.8,
              fillcolor=colors["comp_bg"], line=dict(color=colors["comp"], width=2),
              layer="below")
fig.add_annotation(x=5, y=8.2, text="<b>Computational</b>",
                   showarrow=False, font=dict(size=20, color="#1e40af"))
fig.add_annotation(x=5, y=7.6, text="What is computed and why",
                   showarrow=False, font=dict(size=12, color="#1e3a5f"))

# Impl → Algo arrow
fig.add_annotation(x=5, y=4.95, ax=5, ay=4.25, arrowhead=3, arrowsize=1.5,
                   arrowwidth=2, arrowcolor=colors["algo"], showarrow=True, text="")
fig.add_annotation(x=5.8, y=4.6,
                   text="I/O function<br>demonstration",
                   showarrow=False, font=dict(size=11, color="#115e59"),
                   bgcolor="rgba(240,253,250,0.9)", bordercolor=colors["algo"],
                   borderwidth=1, borderpad=4)

# Algo → Comp arrow
fig.add_annotation(x=5, y=7.15, ax=5, ay=6.45, arrowhead=3, arrowsize=1.5,
                   arrowwidth=2, arrowcolor=colors["comp"], showarrow=True, text="")
fig.add_annotation(x=5.8, y=6.8,
                   text="Generalization +<br>edge-case coverage",
                   showarrow=False, font=dict(size=11, color="#1e40af"),
                   bgcolor="rgba(239,246,255,0.9)", bordercolor=colors["comp"],
                   borderwidth=1, borderpad=4)

# Common drift arrow (curved)
fig.add_annotation(x=1.0, y=8.5, ax=1.0, ay=1.5,
                   arrowhead=3, arrowsize=1.5, arrowwidth=2.5,
                   arrowcolor=colors["drift"], showarrow=True, text="",
                   axref="x", ayref="y",
                   standoff=0)
fig.add_annotation(x=0.5, y=5.0,
                   text="<b>common<br>drift</b>",
                   showarrow=False, font=dict(size=13, color=colors["drift"]),
                   textangle=-90)

# Subtitle
fig.add_annotation(x=5, y=0.3,
                   text="<i>Where most MI circuit claims actually operate</i>",
                   showarrow=False, font=dict(size=12, color="#92400e"))

fig.write_html(str(OUT / "marr_plotly.html"), include_plotlyjs="cdn")
fig.write_image(str(OUT / "marr_plotly.png"), scale=2)
print(f"Written to {OUT / 'marr_plotly.html'} and .png")
