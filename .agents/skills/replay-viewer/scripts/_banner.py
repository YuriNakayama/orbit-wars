"""Inject an agent-identification banner into the Kaggle replay HTML.

The official Kaggle player labels seats as "Player 1 / Player 2 / ..." with no
hint of which agent is which. This module prepends a fixed-position banner to
the HTML that maps each seat to its agent name, version, score and marks the
winner. The banner uses the same player colors as the Orbit Wars renderer
(P1 = magenta, P2 = teal, P3 = orange, P4 = blue) so users can match it to
fleet/planet colors at a glance.
"""

from __future__ import annotations

# Match the palette used by the kaggle_environments orbit_wars renderer.
PLAYER_COLORS: tuple[str, ...] = (
    "#d946ef",  # P1 magenta
    "#14b8a6",  # P2 teal
    "#f59e0b",  # P3 orange
    "#3b82f6",  # P4 blue
)


def _seat_html(idx: int, agent: dict[str, object], is_winner: bool) -> str:
    color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]
    name = str(agent.get("name", f"player {idx}"))
    version = str(agent.get("version", ""))
    score = agent.get("score", "")
    crown = " 👑" if is_winner else ""
    version_html = (
        f'<span style="opacity:0.7;font-size:11px;margin-left:4px;">@{version}</span>'
        if version
        else ""
    )
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;'
        f'border-radius:6px;background:{color};color:#fff;font-weight:600;'
        f'font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px;">'
        f'<span style="opacity:0.7;">P{idx + 1}</span>'
        f'<span>{name}</span>'
        f'{version_html}'
        f'<span style="opacity:0.85;">score={score}</span>'
        f'{crown}'
        f'</span>'
    )


def build_banner(
    match_id: str,
    mode: str,
    seed: int,
    source: str,
    winner_idx: int,
    agents: list[dict[str, object]],
    note: str | None = None,
) -> str:
    """Return a self-contained HTML snippet to prepend to the replay HTML."""
    seats = "".join(_seat_html(i, a, i == winner_idx) for i, a in enumerate(agents))
    note_html = (
        f'<div style="margin-top:6px;color:#fbbf24;font-size:12px;">{note}</div>'
        if note
        else ""
    )
    meta = (
        f'<span style="opacity:0.7;">{source}</span> · '
        f'<span style="opacity:0.7;">{mode}</span> · '
        f'<span style="opacity:0.7;">seed={seed}</span> · '
        f'<code style="opacity:0.85;font-size:11px;">{match_id}</code>'
    )
    return (
        '<div id="replay-viewer-banner" style="'
        "position:sticky;top:0;z-index:9999;"
        "background:#0f172a;color:#e2e8f0;"
        "padding:10px 14px;border-bottom:1px solid #1e293b;"
        "font-family:ui-sans-serif,system-ui,sans-serif;"
        '">'
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;">{seats}</div>'
        f'<div style="margin-top:6px;font-size:12px;">{meta}</div>'
        f"{note_html}"
        "</div>"
    )


def inject(html: str, banner: str) -> str:
    """Prepend the banner to the rendered Kaggle player HTML.

    The Kaggle player HTML starts with an Apache 2.0 license comment and then
    a `<!doctype html>` line. Inserting after `<body>` keeps the document
    well-formed and the banner naturally pins to the top via position:sticky.
    """
    lower = html.lower()
    body_open = lower.find("<body")
    if body_open == -1:
        # No <body> tag — fall back to prepending the banner.
        return banner + html
    close = html.find(">", body_open)
    if close == -1:
        return banner + html
    return html[: close + 1] + banner + html[close + 1 :]
