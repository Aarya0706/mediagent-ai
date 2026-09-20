"""
ui/components.py
-------------------
Reusable Streamlit rendering helpers - pieces of UI used by more than
one tab/screen, as opposed to a single tab's own body. Starts with
render_landing_stat_cards, extracted verbatim from app.py.
"""

import streamlit as st


def render_landing_stat_cards(items):
    """Render a row of small bordered stat cards (icon, label, value, accent
    color) instead of bare st.metric() calls. st.metric() on its own has no
    background or grouping, so a row of them reads as loose numbers floating
    on the page; wrapping each one in a white card with a colored top border
    ties them together as a single glanceable panel."""
    # IMPORTANT: this HTML must stay on one line with no leading whitespace.
    # st.markdown runs its content through a Markdown parser before handing
    # raw-HTML blocks through - a 4+ space indent or a blank line inside the
    # string gets read as a Markdown "indented code block", which is exactly
    # what happened before: only the first <div> rendered as HTML, and
    # everything after the first blank line printed out as literal text.
    card_template = (
        '<div style="background:#FFFFFF;border-radius:14px;padding:14px 16px;'
        'border-top:4px solid {color};box-shadow:0 3px 10px rgba(0,0,0,.07);'
        'flex:1 1 150px;min-width:150px;">'
        '<div style="font-size:12px;font-weight:700;color:#7F8C8D;letter-spacing:.3px;">{icon} {label}</div>'
        '<div style="margin-top:6px;font-size:21px;font-weight:800;color:#2C3E50;">{value}</div>'
        '</div>'
    )
    cards_html = "".join(
        card_template.format(color=color, icon=icon, label=label.upper(), value=value)
        for icon, label, value, color in items
    )
    st.markdown(
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 16px 0;">{cards_html}</div>',
        unsafe_allow_html=True,
    )
