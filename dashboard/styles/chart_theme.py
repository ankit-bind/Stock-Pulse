import plotly.graph_objects as go

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="JetBrains Mono", color="#e2e8f0"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
    hoverlabel=dict(bgcolor="#1a2235", bordercolor="#00d4ff", font_family="JetBrains Mono"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=0, r=0, t=36, b=0),
)

def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the global dark neon theme to any Plotly figure."""
    fig.update_layout(**LAYOUT_BASE)
    
    # Also update any secondary y-axes that might exist
    fig.update_layout(
        yaxis2=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
        yaxis3=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
        yaxis4=dict(gridcolor="rgba(255,255,255,0.05)", showline=False)
    )
    return fig
