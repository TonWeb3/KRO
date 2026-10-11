"""High-Impact Macro Economic News Engine for Gold & Silver.

Tracks and schedules major market-moving macroeconomic releases:
- US CPI (Consumer Price Index)
- US NFP (Non-Farm Payrolls)
- FOMC Interest Rate Decisions & Press Conferences
- US PPI, Retail Sales, and Initial Jobless Claims

Features:
1. Pre-News Preparation Window (10m before release): Alerts bot, tightens risk.
2. News Release Momentum Window (0 to 15m after release): Activates the
   Kronos News Breakout Momentum Strategy to trade explosive directional surges.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    name: str
    currency: str              # 'USD'
    impact: str                # 'HIGH'
    date_str: str              # YYYY-MM-DD
    time_str: str              # HH:MM UTC
    timestamp: float           # Epoch timestamp
    forecast: Optional[str] = ""
    previous: Optional[str] = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "currency": self.currency,
            "impact": self.impact,
            "date_str": self.date_str,
            "time_str": self.time_str,
            "timestamp": self.timestamp,
            "forecast": self.forecast,
            "previous": self.previous,
        }


@dataclass
class NewsState:
    mode: str                  # 'NORMAL', 'PRE_NEWS_ALERT', 'NEWS_BREAKOUT_WINDOW'
    active_event: Optional[NewsEvent]
    next_event: Optional[NewsEvent]
    minutes_until_event: Optional[float]
    minutes_since_event: Optional[float]
    allows_news_breakout: bool
    news_calendar: List[NewsEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "active_event": self.active_event.to_dict() if self.active_event else None,
            "next_event": self.next_event.to_dict() if self.next_event else None,
            "minutes_until_event": round(self.minutes_until_event, 1) if self.minutes_until_event is not None else None,
            "minutes_since_event": round(self.minutes_since_event, 1) if self.minutes_since_event is not None else None,
            "allows_news_breakout": self.allows_news_breakout,
            "upcoming_events": [ev.to_dict() for ev in self.news_calendar[:5]],
        }


class NewsEngine:
    """Manages high-impact macroeconomic event scheduling and breakout detection."""

    def __init__(self, pre_window_min: float = 10.0, post_window_min: float = 15.0):
        self.pre_window_min = pre_window_min
        self.post_window_min = post_window_min
        self.events: List[NewsEvent] = []
        self._seed_recurring_events()

    def _seed_recurring_events(self) -> None:
        """Seeds typical high-impact recurring US news releases for current and upcoming days."""
        now = datetime.datetime.now(datetime.timezone.utc)
        today = now.date()

        # Seed key recurring market events around typical release times (12:30 UTC and 18:00 UTC)
        sample_schedule = [
            ("US CPI Inflation YoY / MoM", "12:30", 0),
            ("US PPI Final Demand", "12:30", 1),
            ("US Non-Farm Payrolls (NFP)", "12:30", 2),
            ("US Retail Sales Advance MoM", "12:30", 3),
            ("FOMC Rate Decision & Statement", "18:00", 4),
            ("Fed Chair Press Conference", "18:30", 4),
            ("US Initial Jobless Claims", "12:30", 5),
        ]

        for name, time_str, day_offset in sample_schedule:
            event_date = today + datetime.timedelta(days=day_offset)
            h, m = map(int, time_str.split(":"))
            dt = datetime.datetime(event_date.year, event_date.month, event_date.day, h, m, tzinfo=datetime.timezone.utc)
            self.events.append(NewsEvent(
                name=name,
                currency="USD",
                impact="HIGH",
                date_str=event_date.isoformat(),
                time_str=time_str + " UTC",
                timestamp=dt.timestamp(),
                forecast="Consensus expected",
                previous="Prior cycle"
            ))

        self.events.sort(key=lambda x: x.timestamp)

    def evaluate(self, current_time: Optional[float] = None) -> NewsState:
        now_ts = current_time or datetime.datetime.now(datetime.timezone.utc).timestamp()

        # Find closest past and future events
        past_events = [e for e in self.events if e.timestamp <= now_ts]
        future_events = [e for e in self.events if e.timestamp > now_ts]

        closest_past = past_events[-1] if past_events else None
        closest_future = future_events[0] if future_events else None

        min_since = (now_ts - closest_past.timestamp) / 60.0 if closest_past else None
        min_until = (closest_future.timestamp - now_ts) / 60.0 if closest_future else None

        # Check if inside News Breakout Momentum Window (0 to 15m after news)
        if min_since is not None and min_since <= self.post_window_min:
            return NewsState(
                mode="NEWS_BREAKOUT_WINDOW",
                active_event=closest_past,
                next_event=closest_future,
                minutes_until_event=min_until,
                minutes_since_event=min_since,
                allows_news_breakout=True,
                news_calendar=future_events
            )

        # Check if inside Pre-News Preparation Window (10m before news)
        if min_until is not None and min_until <= self.pre_window_min:
            return NewsState(
                mode="PRE_NEWS_ALERT",
                active_event=closest_future,
                next_event=closest_future,
                minutes_until_event=min_until,
                minutes_since_event=min_since,
                allows_news_breakout=False,
                news_calendar=future_events
            )

        return NewsState(
            mode="NORMAL",
            active_event=None,
            next_event=closest_future,
            minutes_until_event=min_until,
            minutes_since_event=min_since,
            allows_news_breakout=False,
            news_calendar=future_events
        )
