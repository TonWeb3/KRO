"""Global Financial Session Clock & Precious Metals Liquidity Engine.

Tracks London, New York, Tokyo, and Sydney trading sessions in UTC.
Identifies peak metals liquidity windows (London / NY Golden Overlap)
and filters low-liquidity market regimes.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SessionInfo:
    current_utc: str
    active_sessions: List[str]
    is_golden_overlap: bool     # London / New York overlap (12:00 - 16:00 UTC)
    liquidity_state: str        # 'PEAK_LIQUIDITY', 'NORMAL', 'LOW_LIQUIDITY'
    allows_breakout_trading: bool
    description: str

    def to_dict(self) -> dict:
        return {
            "current_utc": self.current_utc,
            "active_sessions": self.active_sessions,
            "is_golden_overlap": self.is_golden_overlap,
            "liquidity_state": self.liquidity_state,
            "allows_breakout_trading": self.allows_breakout_trading,
            "description": self.description,
        }


def get_session_info(now_utc: datetime.datetime | None = None) -> SessionInfo:
    """Calculates active market sessions and liquidity conditions for Gold/Silver."""
    now = now_utc or datetime.datetime.now(datetime.timezone.utc)
    hour = now.hour + (now.minute / 60.0)
    weekday = now.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday

    active: List[str] = []

    # Weekend check
    if weekday == 5 or (weekday == 4 and hour >= 21.0) or (weekday == 6 and hour < 21.0):
        return SessionInfo(
            current_utc=now.strftime("%H:%M:%S UTC"),
            active_sessions=["WEEKEND_CLOSED"],
            is_golden_overlap=False,
            liquidity_state="CLOSED",
            allows_breakout_trading=False,
            description="Market closed for weekend. No trading permitted.",
        )

    # 1. London Session: 07:00 - 16:00 UTC
    if 7.0 <= hour < 16.0:
        active.append("London")

    # 2. New York Session: 12:00 - 21:00 UTC
    if 12.0 <= hour < 21.0:
        active.append("New York")

    # 3. Tokyo / Asian Session: 23:00 - 07:00 UTC
    if hour >= 23.0 or hour < 7.0:
        active.append("Tokyo")

    # 4. Sydney Session: 21:00 - 05:00 UTC
    if hour >= 21.0 or hour < 5.0:
        active.append("Sydney")

    # London / New York Golden Overlap: 12:00 - 16:00 UTC (Peak Metals Momentum)
    is_golden = (12.0 <= hour < 16.0)

    if is_golden:
        liq = "PEAK_LIQUIDITY"
        desc = "London + New York Golden Overlap: Maximum institutional volume for Gold & Silver."
        allow_breakout = True
    elif "London" in active or "New York" in active:
        liq = "NORMAL"
        desc = "European / US active session: Standard liquid trading conditions."
        allow_breakout = True
    else:
        liq = "LOW_LIQUIDITY"
        desc = "Asian Session: Lower volatility range-bound conditions. S/R bounces preferred."
        allow_breakout = False

    return SessionInfo(
        current_utc=now.strftime("%H:%M:%S UTC"),
        active_sessions=active,
        is_golden_overlap=is_golden,
        liquidity_state=liq,
        allows_breakout_trading=allow_breakout,
        description=desc,
    )
