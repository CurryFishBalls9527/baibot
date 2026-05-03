"""Overlay vocabulary — the wire contract between extractors and the chart.

A ``ChartPayload`` is what every strategy returns from its extractor.
The frontend renders this dict directly via lightweight-charts.

Vocabulary kinds:
  - ``line``   straight segment between two (ts, price) points
  - ``zone``   rectangle [from_t, to_t] × [low, high]
  - ``marker`` point on a bar with a label and side hint
  - ``ma``     moving-average series of (ts, value) tuples
  - ``level``  horizontal price line, optional time range
  - ``band``   like zone but visually filled (used for VCP base, OR range)

Times are unix seconds (UTC). Prices are floats. The frontend treats
unknown ``style`` strings as defaults.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


# ── Bars ─────────────────────────────────────────────────────────────


class Bar(TypedDict):
    time: int          # unix seconds, UTC
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]


# ── Overlay primitives ───────────────────────────────────────────────


class LineOverlay(TypedDict):
    kind: Literal["line"]
    from_t: int
    from_p: float
    to_t: int
    to_p: float
    label: Optional[str]
    style: Optional[str]    # "solid" | "dashed" | "dotted"
    color: Optional[str]
    width: Optional[int]


class ZoneOverlay(TypedDict):
    kind: Literal["zone"]
    from_t: int
    to_t: int
    low: float
    high: float
    label: Optional[str]
    color: Optional[str]


class MarkerOverlay(TypedDict):
    kind: Literal["marker"]
    time: int
    price: float
    label: Optional[str]
    side: Optional[Literal["buy", "sell", "info"]]
    color: Optional[str]


class MAOverlay(TypedDict):
    kind: Literal["ma"]
    period: int
    values: List[List[float]]   # [[ts, value], ...]
    color: Optional[str]
    label: Optional[str]


class LevelOverlay(TypedDict):
    kind: Literal["level"]
    price: float
    from_t: Optional[int]
    to_t: Optional[int]
    label: Optional[str]
    style: Optional[str]
    color: Optional[str]


class BandOverlay(TypedDict):
    kind: Literal["band"]
    from_t: int
    to_t: int
    low: float
    high: float
    label: Optional[str]
    color: Optional[str]


# ── Reasoning panel ──────────────────────────────────────────────────


class Criterion(TypedDict):
    name: str
    passed: bool
    value: Optional[Any]


class Metric(TypedDict):
    label: str
    value: Any


class Reasoning(TypedDict):
    headline: str               # one-line trade thesis
    criteria: List[Criterion]   # ✓/✗ rows
    metrics: List[Metric]       # key/value grid
    narrative: Optional[str]    # collapsible long-form (LLM debate, etc.)


# ── Trade fills (rendered as buy/sell markers always) ────────────────


class Fill(TypedDict):
    time: int
    price: float
    side: Literal["buy", "sell"]
    qty: float
    reasoning: Optional[str]    # from the trades.reasoning column


# ── Top-level payload ────────────────────────────────────────────────


class ChartPayload(TypedDict):
    symbol: str
    variant: str
    strategy_type: str
    timeframe: str              # "30m" | "1d" | "15m" | "5m"
    bars: List[Bar]
    overlays: List[Dict]        # any of the *Overlay TypedDicts above
    fills: List[Fill]
    reasoning: Reasoning
    error: Optional[str]
    is_open: bool               # True if no SELL fill exists for this trade
