"""UI 样式配置 - 暗黑/明亮模式 CSS 常量"""

DARK_MODE_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stApp .stMarkdown, .stApp .stText { color: #fafafa; }
    .stApp .stMetric { background-color: #1a1a2e; border-radius: 8px; padding: 10px; }
    .stApp section[data-testid="stExpander"] { background-color: #1a1a2e; }
    .stApp .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stApp .stTabs [data-baseweb="tab"] { background-color: #1a1a2e; color: #aaa; }
    .stApp .stTabs [aria-selected="true"] { background-color: #4a4a6a !important; color: #fff !important; }
</style>
"""

LIGHT_MODE_CSS = """
<style>
    .main-title { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { text-align: center; color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .risk-badge { display: inline-block; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; }
</style>
"""


def apply_theme(dark: bool):
    """应用暗黑/明亮模式样式"""
    import streamlit as st
    st.markdown(DARK_MODE_CSS if dark else LIGHT_MODE_CSS, unsafe_allow_html=True)
