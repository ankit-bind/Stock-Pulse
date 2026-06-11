# dashboard/styles/tokens.py — inject once at app startup

CSS_TOKENS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
--sp-bg: #0a0e1a; /* deep navy canvas */
--sp-surface: #111827; /* card surface */
--sp-surface2: #1a2235; /* raised elements */
--sp-border: rgba(255,255,255,0.07);
--sp-accent: #00d4ff; /* cyan - primary action */
--sp-accent2: #7c3aed; /* violet - secondary */
--sp-green: #10b981; /* gain */
--sp-red: #ef4444; /* loss */
--sp-amber: #f59e0b; /* warning / RSI */
--sp-text: #e2e8f0;
--sp-text-muted: #94a3b8;
--sp-font: 'JetBrains Mono', 'IBM Plex Mono', monospace;
--sp-font-prose: 'DM Sans', sans-serif;
--sp-radius: 10px;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--sp-bg) !important;
    color: var(--sp-text) !important;
    font-family: var(--sp-font-prose) !important;
}

h1, h2, h3, h4, h5, h6, p, span, div {
    font-family: var(--sp-font-prose);
}

code, pre {
    font-family: var(--sp-font) !important;
}

/* KPI Card Shimmer Sweep Update (Neon Cards v2) */
.neon-card {
  position: relative;
  background: var(--sp-surface);
  border-radius: var(--sp-radius);
  height: 110px;
  min-height: 110px;
  max-height: 110px;
  overflow: hidden;
  box-sizing: border-box;
  margin-bottom: 1rem;
  border: 1px solid var(--sp-border);
  transition: all 0.3s ease;
}

.neon-card:hover {
  transform: translateY(-2px);
  border-color: var(--sp-accent);
  box-shadow: 0 4px 15px rgba(0, 212, 255, 0.15);
}

.neon-card-inner {
  background: var(--sp-surface);
  position: absolute;
  inset: 0;
  padding: 10px 15px;
  box-sizing: border-box;
  z-index: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.neon-card-title {
  font-family: var(--sp-font-prose);
  font-size: 0.9rem;
  color: var(--sp-text-muted);
  margin-bottom: 5px;
  font-weight: 600;
}

.neon-card-value {
  font-family: var(--sp-font);
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--sp-text);
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Layout upgrades - Sticky Sidebar & Tabs */
section[data-testid="stSidebar"] {
  position: sticky; top: 0; height: 100vh; overflow-y: auto;
  background: var(--sp-surface) !important;
  border-right: 1px solid var(--sp-border);
  scrollbar-width: thin; scrollbar-color: var(--sp-accent) transparent;
}

.stTabs [data-baseweb="tab"] {
  font-family: var(--sp-font); font-size: 13px;
  border-bottom: 2px solid transparent; transition: border-color .2s;
}

.stTabs [aria-selected="true"] {
  border-bottom-color: var(--sp-accent);
  color: var(--sp-accent) !important;
}

/* KPI Responsive Grid */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}
/* Custom section divider */
.gradient-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--sp-accent), transparent);
  margin: 1.5rem 0;
  border: none;
}
</style>
"""
