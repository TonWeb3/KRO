"""LuxAlgo Support and Resistance Levels with Breaks & Candle Zones Engine.

Direct Python/NumPy implementation of:
LuxAlgo - Support and Resistance Levels with Breaks (Pine Script v4):
- pivothigh(leftBars=15, rightBars=15) & pivotlow(leftBars=15, rightBars=15)
- Support & Resistance Candle Zones: marks both the high and low of the pivot candle
  as a persistent dynamic zone [zone_floor, zone_ceiling]
- Independent per-timeframe S/R zone computation (5m, 15m, 30m, 1h, 4h)
- Adaptive Balance-Driven Timeframe Selection:
    * Small balance (< $250): Focuses strictly on 5m micro-zones with ultra-tight SL
    * Medium balance ($250 - $750): Trades 15m intermediate structure setups
    * Large balance (> $750): Trades 30m / 1h macro structure with wider swing capacity
- Multi-Timeframe Confluence for SL & TP:
    * Stop Loss anchored to entry timeframe zone floor/ceiling for tightest risk
    * Take Profit targets higher-timeframe resistance/support lines (15m, 30m, 1h, 4h)
- Volume % Oscillator: 100 * (ema(vol, 5) - ema(vol, 10)) / ema(vol, 10)
- Candle-in-zone entry detection: tracks when spot or candle penetrates the zone
- Breakout Detection ('BREAK_UP' / 'BREAK_DOWN') with candle body & volume confirmation
- Reversal Detection ('REVERSAL_UP' / 'REVERSAL_DOWN') with rejection wicks & candle turn
- Kronos Model Integration: AI model detects breaks vs reversals from the zones
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class SRZone:
    zone_type: str         # 'RESISTANCE' or 'SUPPORT'
    high: float            # Ceiling of the pivot candle (pivot high)
    low: float             # Floor of the pivot candle (pivot low)
    candle_open: float
    candle_close: float
    timeframe: str         # '5m', '15m', '30m', '1h', '4h'
    pivot_bar_index: int
    timestamp: str = ""
    status: str = "ACTIVE" # 'ACTIVE', 'TESTING', 'BROKEN_UP', 'BROKEN_DOWN', 'REVERSED_UP', 'REVERSED_DOWN'
    touches: int = 1

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def height(self) -> float:
        return max(1e-4, self.high - self.low)

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def to_dict(self) -> dict:
        return {
            "zone_type": self.zone_type,
            "high": round(self.high, 3),
            "low": round(self.low, 3),
            "mid": round(self.mid, 3),
            "candle_open": round(self.candle_open, 3),
            "candle_close": round(self.candle_close, 3),
            "timeframe": self.timeframe,
            "pivot_bar_index": self.pivot_bar_index,
            "timestamp": self.timestamp,
            "status": self.status,
            "touches": self.touches,
        }


@dataclass
class LuxBreakEvent:
    bar_index: int
    timestamp: str
    price: float
    break_type: str        # 'BREAK_UP', 'BREAK_DOWN', 'REVERSAL_UP', 'REVERSAL_DOWN', 'BULL_WICK', 'BEAR_WICK'
    label: str             # 'B', 'REV ▲', 'REV ▼', 'Bull Wick', 'Bear Wick'
    volume_osc: float

    def to_dict(self) -> dict:
        return {
            "bar_index": self.bar_index,
            "timestamp": self.timestamp,
            "price": round(self.price, 3),
            "break_type": self.break_type,
            "label": self.label,
            "volume_osc": round(self.volume_osc, 1),
        }


@dataclass
class SRLevel:
    price: float
    level_type: str        # 'SUPPORT' or 'RESISTANCE'
    strength: int          # 1 to 5
    source: str            # 'LUX_ACTIVE', 'LUX_PIVOT', 'KRONOS_BAND'
    timeframe: str         # '5m', '15m', '30m', '1h', '4h'
    touches: int = 1
    last_touch_ts: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "price": round(self.price, 3),
            "level_type": self.level_type,
            "strength": self.strength,
            "source": self.source,
            "timeframe": self.timeframe,
            "touches": self.touches,
        }


@dataclass
class SRContext:
    symbol: str
    spot: float
    active_resistance: Optional[float]     # Ceiling of active resistance zone
    active_support: Optional[float]        # Floor of active support zone
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    dist_to_support_pct: float
    dist_to_resistance_pct: float
    zone: str                              # 'IN_SUPPORT_ZONE', 'IN_RESISTANCE_ZONE', 'AT_SUPPORT', 'AT_RESISTANCE', 'BREAKOUT_UP', 'BREAKOUT_DOWN', 'REVERSAL_UP', 'REVERSAL_DOWN', 'MID_RANGE'
    volume_osc: float = 0.0
    latest_break: Optional[str] = None     # 'BREAK_UP', 'BREAK_DOWN', 'REVERSAL_UP', 'REVERSAL_DOWN', 'BULL_WICK', 'BEAR_WICK'
    broken_level: Optional[float] = None
    all_levels: List[SRLevel] = field(default_factory=list)
    break_events: List[LuxBreakEvent] = field(default_factory=list)
    active_res_zone: Optional[SRZone] = None
    active_sup_zone: Optional[SRZone] = None
    all_zones: List[SRZone] = field(default_factory=list)
    model_zone_verdict: str = "NONE"       # 'ZONE_BREAK_UP', 'ZONE_REVERSAL_DOWN', 'ZONE_BREAK_DOWN', 'ZONE_REVERSAL_UP', 'NONE'
    model_zone_reason: str = ""
    # Multi-Timeframe & Balance-Adaptive Enhancements
    entry_sr_timeframe: str = "15m"        # Timeframe used for entry ('5m', '15m', '30m', '1h')
    balance_bracket: str = "MID_BALANCE"   # 'SMALL_BALANCE', 'MID_BALANCE', 'LARGE_BALANCE'
    by_timeframe: Dict[str, Any] = field(default_factory=dict)
    higher_tf_targets: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "spot": round(self.spot, 3),
            "active_resistance": round(self.active_resistance, 3) if self.active_resistance else None,
            "active_support": round(self.active_support, 3) if self.active_support else None,
            "nearest_support": round(self.nearest_support, 3) if self.nearest_support else None,
            "nearest_resistance": round(self.nearest_resistance, 3) if self.nearest_resistance else None,
            "dist_to_support_pct": round(self.dist_to_support_pct, 3),
            "dist_to_resistance_pct": round(self.dist_to_resistance_pct, 3),
            "zone": self.zone,
            "volume_osc": round(self.volume_osc, 1),
            "latest_break": self.latest_break,
            "broken_level": round(self.broken_level, 3) if self.broken_level else None,
            "levels": [lvl.to_dict() for lvl in self.all_levels],
            "break_events": [b.to_dict() for b in self.break_events[-15:]],
            "active_resistance_zone": self.active_res_zone.to_dict() if self.active_res_zone else None,
            "active_support_zone": self.active_sup_zone.to_dict() if self.active_sup_zone else None,
            "all_zones": [z.to_dict() for z in self.all_zones[-6:]],
            "model_zone_verdict": self.model_zone_verdict,
            "model_zone_reason": self.model_zone_reason,
            "entry_sr_timeframe": self.entry_sr_timeframe,
            "balance_bracket": self.balance_bracket,
            "by_timeframe": self.by_timeframe,
            "higher_tf_targets": self.higher_tf_targets,
        }


def compute_lux_pivots_and_breaks(
    df: pd.DataFrame,
    left_bars: int = 15,
    right_bars: int = 15,
    volume_thresh: float = 20.0,
    timeframe: str = "15m"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[LuxBreakEvent], Optional[SRZone], Optional[SRZone], List[SRZone]]:
    """Calculates LuxAlgo Support & Resistance Candle Zones with Breaks & Reversals.

    Marks both HIGH and LOW of pivot candles as full support/resistance zones.
    Detects when subsequent candles enter the zone, and identifies breaks or reversals.
    """
    n = len(df)
    if n < left_bars + right_bars + 2:
        empty = np.full(n, np.nan)
        return empty, empty, np.zeros(n), [], None, None, []

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    volumes = df["volume"].to_numpy(dtype=float)
    timestamps = [str(t) for t in df["timestamps"]] if "timestamps" in df.columns else ["" for _ in range(n)]

    # 1. Volume % Oscillator: short = ema(vol, 5), long = ema(vol, 10), osc = 100 * (short - long) / long
    vol_series = pd.Series(volumes)
    if vol_series.sum() == 0:
        vol_series = pd.Series(np.maximum(1e-4, highs - lows))
    short_v = vol_series.ewm(span=5, adjust=False).mean().to_numpy()
    long_v = vol_series.ewm(span=10, adjust=False).mean().to_numpy()
    osc = np.where(long_v > 0, 100.0 * (short_v - long_v) / long_v, 0.0)

    # 2. Pivot High and Pivot Low arrays
    high_use_pivot = np.full(n, np.nan)
    low_use_pivot = np.full(n, np.nan)

    last_hp = np.nan
    last_lp = np.nan

    active_res_zone: Optional[SRZone] = None
    active_sup_zone: Optional[SRZone] = None
    all_zones: List[SRZone] = []
    breaks: List[LuxBreakEvent] = []

    # Slide across bars
    for i in range(left_bars + right_bars, n):
        p_idx = i - right_bars
        target_h = highs[p_idx]
        target_l = lows[p_idx]
        target_o = opens[p_idx]
        target_c = closes[p_idx]
        target_ts = timestamps[p_idx]

        # Pivot High test: target_h strictly > left bars and >= right bars
        left_h = highs[p_idx - left_bars : p_idx]
        right_h = highs[p_idx + 1 : p_idx + right_bars + 1]
        if len(left_h) and len(right_h):
            if target_h > np.max(left_h) and target_h >= np.max(right_h):
                last_hp = target_h
                # Mark BOTH the high and low of the pivot candle as the RESISTANCE ZONE
                active_res_zone = SRZone(
                    zone_type="RESISTANCE",
                    high=target_h,
                    low=target_l,
                    candle_open=target_o,
                    candle_close=target_c,
                    timeframe=timeframe,
                    pivot_bar_index=p_idx,
                    timestamp=target_ts,
                    status="ACTIVE",
                    touches=1
                )
                all_zones.append(active_res_zone)

        # Pivot Low test: target_l strictly < left bars and <= right bars
        left_l = lows[p_idx - left_bars : p_idx]
        right_l = lows[p_idx + 1 : p_idx + right_bars + 1]
        if len(left_l) and len(right_l):
            if target_l < np.min(left_l) and target_l <= np.min(right_l):
                last_lp = target_l
                # Mark BOTH the high and low of the pivot candle as the SUPPORT ZONE
                active_sup_zone = SRZone(
                    zone_type="SUPPORT",
                    high=target_h,
                    low=target_l,
                    candle_open=target_o,
                    candle_close=target_c,
                    timeframe=timeframe,
                    pivot_bar_index=p_idx,
                    timestamp=target_ts,
                    status="ACTIVE",
                    touches=1
                )
                all_zones.append(active_sup_zone)

        high_use_pivot[i] = last_hp
        low_use_pivot[i] = last_lp

        # Check zone interactions & breaks on bar i
        c_prev = closes[i - 1]
        c_curr = closes[i]
        o_curr = opens[i]
        h_curr = highs[i]
        l_curr = lows[i]
        osc_curr = osc[i]
        bar_ts = timestamps[i]

        # ---------------- RESISTANCE ZONE INTERACTIONS ----------------
        if active_res_zone is not None:
            # Did price enter the resistance zone?
            entered_res = (h_curr >= active_res_zone.low and l_curr <= active_res_zone.high)
            if entered_res:
                active_res_zone.touches += 1
                active_res_zone.status = "TESTING"

                # Check for REVERSAL DOWN from Resistance Zone
                upper_wick = h_curr - max(o_curr, c_curr)
                candle_range = max(1e-4, h_curr - l_curr)
                is_bearish = c_curr < o_curr
                wick_ratio = upper_wick / candle_range

                if is_bearish and (wick_ratio >= 0.25 or c_curr <= active_res_zone.mid):
                    breaks.append(LuxBreakEvent(
                        bar_index=i, timestamp=bar_ts, price=c_curr,
                        break_type="REVERSAL_DOWN", label="REV ▼", volume_osc=osc_curr
                    ))
                    active_res_zone.status = "REVERSED_DOWN"

            # Check for BREAKOUT UP above Resistance Zone
            crossed_above_ceiling = (c_prev <= active_res_zone.high and c_curr > active_res_zone.high)
            if crossed_above_ceiling:
                is_adverse_wick = (o_curr - l_curr) > (c_curr - o_curr)
                if not is_adverse_wick and (osc_curr > volume_thresh or c_curr > o_curr):
                    breaks.append(LuxBreakEvent(
                        bar_index=i, timestamp=bar_ts, price=c_curr,
                        break_type="BREAK_UP", label="B", volume_osc=osc_curr
                    ))
                    active_res_zone.status = "BROKEN_UP"
                elif is_adverse_wick:
                    breaks.append(LuxBreakEvent(
                        bar_index=i, timestamp=bar_ts, price=c_curr,
                        break_type="BULL_WICK", label="Bull Wick", volume_osc=osc_curr
                    ))

        # ---------------- SUPPORT ZONE INTERACTIONS ----------------
        if active_sup_zone is not None:
            # Did price enter the support zone?
            entered_sup = (l_curr <= active_sup_zone.high and h_curr >= active_sup_zone.low)
            if entered_sup:
                active_sup_zone.touches += 1
                active_sup_zone.status = "TESTING"

                # Check for REVERSAL UP from Support Zone
                lower_wick = min(o_curr, c_curr) - l_curr
                candle_range = max(1e-4, h_curr - l_curr)
                is_bullish = c_curr > o_curr
                wick_ratio = lower_wick / candle_range

                if is_bullish and (wick_ratio >= 0.25 or c_curr >= active_sup_zone.mid):
                    breaks.append(LuxBreakEvent(
                        bar_index=i, timestamp=bar_ts, price=c_curr,
                        break_type="REVERSAL_UP", label="REV ▲", volume_osc=osc_curr
                    ))
                    active_sup_zone.status = "REVERSED_UP"

            # Check for BREAKDOWN DOWN below Support Zone
            crossed_below_floor = (c_prev >= active_sup_zone.low and c_curr < active_sup_zone.low)
            if crossed_below_floor:
                is_adverse_wick = (o_curr - c_curr) < (h_curr - o_curr)
                if not is_adverse_wick and (osc_curr > volume_thresh or c_curr < o_curr):
                    breaks.append(LuxBreakEvent(
                        bar_index=i, timestamp=bar_ts, price=c_curr,
                        break_type="BREAK_DOWN", label="B", volume_osc=osc_curr
                    ))
                    active_sup_zone.status = "BROKEN_DOWN"
                elif is_adverse_wick:
                    breaks.append(LuxBreakEvent(
                        bar_index=i, timestamp=bar_ts, price=c_curr,
                        break_type="BEAR_WICK", label="Bear Wick", volume_osc=osc_curr
                    ))

    # Evaluate all past and present zones across the entire 2000-candle history:
    cur_spot = closes[-1] if n > 0 else 0.0
    cur_high = highs[-1] if n > 0 else cur_spot
    cur_low = lows[-1] if n > 0 else cur_spot

    # 1. Did current candle enter ANY past or present Resistance Zone (either body inside or wick high pierced)?
    containing_res = [z for z in all_zones if z.zone_type == "RESISTANCE" and (z.low <= cur_spot <= z.high or (cur_high >= z.low and cur_low <= z.high))]
    if containing_res:
        active_res_zone = containing_res[-1]
    else:
        overhead_res = [z for z in all_zones if z.zone_type == "RESISTANCE" and z.high >= cur_spot and z.status != "BROKEN_UP"]
        if overhead_res:
            active_res_zone = min(overhead_res, key=lambda z: (z.low - cur_spot) if z.low >= cur_spot else 0.0)

    # 2. Did current candle enter ANY past or present Support Zone (either body inside or wick low swept)?
    containing_sup = [z for z in all_zones if z.zone_type == "SUPPORT" and (z.low <= cur_spot <= z.high or (cur_low <= z.high and cur_high >= z.low))]
    if containing_sup:
        active_sup_zone = containing_sup[-1]
    else:
        underfoot_sup = [z for z in all_zones if z.zone_type == "SUPPORT" and z.low <= cur_spot and z.status != "BROKEN_DOWN"]
        if underfoot_sup:
            active_sup_zone = min(underfoot_sup, key=lambda z: (cur_spot - z.high) if cur_spot >= z.high else 0.0)

    return high_use_pivot, low_use_pivot, osc, breaks, active_res_zone, active_sup_zone, all_zones


def compute_sr_context(
    symbol: str,
    spot: float,
    df_5m: Optional[pd.DataFrame] = None,
    df_15m: Optional[pd.DataFrame] = None,
    df_30m: Optional[pd.DataFrame] = None,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    fc_lo: Optional[float] = None,
    fc_hi: Optional[float] = None,
    entry_fc: Optional[Any] = None,
    balance: Optional[float] = None,
    entry_timeframe: Optional[str] = None,
) -> SRContext:
    """Computes independent LuxAlgo S/R Candle Zones for each timeframe and adaptively selects entry based on balance.

    User Directive:
    - Each timeframe (5m, 15m, 30m, 1h, 4h) maintains its own Support and Resistance Candle Zones.
    - Small balance (< $250): Uses 5m micro-zones with ultra-tight SL to protect capital.
    - Mid balance ($250 - $750): Uses 15m structure setups.
    - Large balance (> $750): Uses 30m / 1h higher S/R zones that can absorb wider swings and bigger runs.
    - Records what timeframe S/R was used, and maps higher-timeframe lines to support Take Profit and Stop Loss.
    """
    bal = balance if (balance is not None and balance > 0) else 100.0

    # 1. Determine balance bracket and target entry timeframe
    if bal < 250.0:
        balance_bracket = "SMALL_BALANCE"
        target_entry_tf = entry_timeframe or "5m"
    elif bal < 750.0:
        balance_bracket = "MID_BALANCE"
        target_entry_tf = entry_timeframe or "15m"
    else:
        balance_bracket = "LARGE_BALANCE"
        target_entry_tf = entry_timeframe or "30m"

    by_timeframe: Dict[str, Any] = {}
    all_levels: List[SRLevel] = []
    all_zones: List[SRZone] = []
    all_break_events: List[LuxBreakEvent] = []

    # 2. Compute full LuxAlgo pivot zones for each timeframe
    tf_configs = [
        ("5m", df_5m, 10, 10, 3),
        ("15m", df_15m, 15, 15, 4),
        ("30m", df_30m, 15, 15, 4),
        ("1h", df_1h, 12, 12, 5),
        ("4h", df_4h, 10, 10, 5),
    ]

    for tf_name, df, left_b, right_b, strength in tf_configs:
        if df is None or len(df) < 30:
            continue
        try:
            hp, lp, osc_arr, breaks, r_zone, s_zone, z_list = compute_lux_pivots_and_breaks(
                df, left_bars=left_b, right_bars=right_b, volume_thresh=20.0, timeframe=tf_name
            )
            all_zones.extend(z_list)
            all_break_events.extend(breaks)

            # Evaluate model interaction for this specific timeframe
            model_verdict = "NONE"
            model_reason = ""
            last_h = float(df["high"].iloc[-1]) if len(df) else spot
            last_l = float(df["low"].iloc[-1]) if len(df) else spot

            entered_by_wick_r = False
            if r_zone:
                tol_r = max(0.2, r_zone.height * 0.35)
                # Candle entered resistance if spot close is inside/near OR if the candle's high pierced into the zone
                entered_by_wick_r = (last_h >= r_zone.low and last_l <= r_zone.high)
                entered_by_spot_r = (spot >= r_zone.low - tol_r) and (spot <= r_zone.high + tol_r)

                if entered_by_spot_r or entered_by_wick_r:
                    if entry_fc is not None:
                        up_p = getattr(entry_fc, "upside_prob", 0.5)
                        if up_p >= 0.52:
                            model_verdict = "ZONE_BREAK_UP"
                            model_reason = f"Breakout above {tf_name} Resistance Zone [${r_zone.low:.2f} - ${r_zone.high:.2f}]"
                        elif up_p <= 0.48:
                            model_verdict = "ZONE_REVERSAL_DOWN"
                            if entered_by_wick_r and spot < r_zone.low:
                                model_reason = f"Bearish Reversal rejection wick from {tf_name} Resistance Zone [${r_zone.low:.2f} - ${r_zone.high:.2f}] (High pierced zone to ${last_h:.2f}, closed outside at ${spot:.2f})"
                            else:
                                model_reason = f"Bearish Reversal from {tf_name} Resistance Zone [${r_zone.low:.2f} - ${r_zone.high:.2f}]"
                        else:
                            model_verdict = "TESTING_RESISTANCE"
                            model_reason = f"Testing {tf_name} Resistance Zone [${r_zone.low:.2f} - ${r_zone.high:.2f}]"
                all_levels.append(SRLevel(price=r_zone.high, level_type="RESISTANCE", strength=strength, source=f"LUX_{tf_name.upper()}", timeframe=tf_name))
                all_levels.append(SRLevel(price=r_zone.low, level_type="RESISTANCE", strength=max(1, strength - 1), source=f"LUX_{tf_name.upper()}", timeframe=tf_name))

            entered_by_wick_s = False
            if s_zone:
                tol_s = max(0.2, s_zone.height * 0.35)
                # Candle entered support if spot close is inside/near OR if the candle's low swept down into the zone
                entered_by_wick_s = (last_l <= s_zone.high and last_h >= s_zone.low)
                entered_by_spot_s = (spot >= s_zone.low - tol_s) and (spot <= s_zone.high + tol_s)

                if entered_by_spot_s or entered_by_wick_s:
                    if entry_fc is not None:
                        up_p = getattr(entry_fc, "upside_prob", 0.5)
                        if up_p >= 0.52:
                            model_verdict = "ZONE_REVERSAL_UP"
                            if entered_by_wick_s and spot > s_zone.high:
                                model_reason = f"Bullish Reversal bounce wick from {tf_name} Support Zone [${s_zone.low:.2f} - ${s_zone.high:.2f}] (Low swept zone to ${last_l:.2f}, closed outside at ${spot:.2f})"
                            else:
                                model_reason = f"Bullish Reversal bounce from {tf_name} Support Zone [${s_zone.low:.2f} - ${s_zone.high:.2f}]"
                        elif up_p <= 0.48:
                            model_verdict = "ZONE_BREAK_DOWN"
                            model_reason = f"Breakdown below {tf_name} Support Zone [${s_zone.low:.2f} - ${s_zone.high:.2f}]"
                        else:
                            model_verdict = "TESTING_SUPPORT"
                            model_reason = f"Testing {tf_name} Support Zone [${s_zone.low:.2f} - ${s_zone.high:.2f}]"
                all_levels.append(SRLevel(price=s_zone.low, level_type="SUPPORT", strength=strength, source=f"LUX_{tf_name.upper()}", timeframe=tf_name))
                all_levels.append(SRLevel(price=s_zone.high, level_type="SUPPORT", strength=max(1, strength - 1), source=f"LUX_{tf_name.upper()}", timeframe=tf_name))

            # Determine timeframe zone position
            dist_s = ((spot - s_zone.low) / spot * 100.0) if s_zone else 999.0
            dist_r = ((r_zone.high - spot) / spot * 100.0) if r_zone else 999.0
            if model_verdict == "ZONE_BREAK_UP":
                tf_z = "BREAKOUT_UP"
            elif model_verdict == "ZONE_BREAK_DOWN":
                tf_z = "BREAKOUT_DOWN"
            elif model_verdict == "ZONE_REVERSAL_UP":
                tf_z = "REVERSAL_UP"
            elif model_verdict == "ZONE_REVERSAL_DOWN":
                tf_z = "REVERSAL_DOWN"
            elif r_zone and (r_zone.contains(spot) or entered_by_wick_r):
                tf_z = "IN_RESISTANCE_ZONE"
            elif s_zone and (s_zone.contains(spot) or entered_by_wick_s):
                tf_z = "IN_SUPPORT_ZONE"
            elif dist_s <= 0.25:
                tf_z = "AT_SUPPORT"
            elif dist_r <= 0.25:
                tf_z = "AT_RESISTANCE"
            else:
                tf_z = "MID_RANGE"

            by_timeframe[tf_name] = {
                "timeframe": tf_name,
                "active_resistance_zone": r_zone.to_dict() if r_zone else None,
                "active_support_zone": s_zone.to_dict() if s_zone else None,
                "active_resistance": round(r_zone.high, 3) if r_zone else None,
                "active_support": round(s_zone.low, 3) if s_zone else None,
                "resistance_floor": round(r_zone.low, 3) if r_zone else None,
                "support_ceiling": round(s_zone.high, 3) if s_zone else None,
                "zone": tf_z,
                "volume_osc": round(float(osc_arr[-1]), 1) if len(osc_arr) else 0.0,
                "model_zone_verdict": model_verdict,
                "model_zone_reason": model_reason,
                "break_events": [b.to_dict() for b in breaks[-10:]],
            }
        except Exception as e:
            log.warning("S/R computation error on %s %s: %s", symbol, tf_name, e)

    # 3. Select primary entry timeframe based on balance
    chosen_tf = target_entry_tf
    if chosen_tf not in by_timeframe or by_timeframe[chosen_tf].get("active_resistance") is None:
        if balance_bracket == "SMALL_BALANCE":
            fallback_order = ["5m", "15m", "30m", "1h", "4h"]
        elif balance_bracket == "MID_BALANCE":
            fallback_order = ["15m", "30m", "5m", "1h", "4h"]
        else:
            fallback_order = ["30m", "1h", "15m", "4h", "5m"]
        for fb in fallback_order:
            if fb in by_timeframe and by_timeframe[fb].get("active_resistance") is not None:
                chosen_tf = fb
                break

    primary_data = by_timeframe.get(chosen_tf, {})

    # Active primary zones
    active_res_dict = primary_data.get("active_resistance_zone")
    active_sup_dict = primary_data.get("active_support_zone")

    active_res_zone = None
    if active_res_dict:
        active_res_zone = SRZone(
            zone_type="RESISTANCE",
            high=active_res_dict["high"],
            low=active_res_dict["low"],
            candle_open=active_res_dict["candle_open"],
            candle_close=active_res_dict["candle_close"],
            timeframe=chosen_tf,
            pivot_bar_index=active_res_dict.get("pivot_bar_index", -1),
            timestamp=active_res_dict.get("timestamp", ""),
            status=active_res_dict.get("status", "ACTIVE")
        )

    active_sup_zone = None
    if active_sup_dict:
        active_sup_zone = SRZone(
            zone_type="SUPPORT",
            high=active_sup_dict["high"],
            low=active_sup_dict["low"],
            candle_open=active_sup_dict["candle_open"],
            candle_close=active_sup_dict["candle_close"],
            timeframe=chosen_tf,
            pivot_bar_index=active_sup_dict.get("pivot_bar_index", -1),
            timestamp=active_sup_dict.get("timestamp", ""),
            status=active_sup_dict.get("status", "ACTIVE")
        )

    active_res = active_res_zone.high if active_res_zone else (primary_data.get("active_resistance") or spot * 1.008)
    active_sup = active_sup_zone.low if active_sup_zone else (primary_data.get("active_support") or spot * 0.992)
    zone = primary_data.get("zone", "MID_RANGE")
    volume_osc = primary_data.get("volume_osc", 0.0)
    model_zone_verdict = primary_data.get("model_zone_verdict", "NONE")
    model_zone_reason = primary_data.get("model_zone_reason", "")
    breaks = primary_data.get("break_events", [])
    latest_break = breaks[-1]["break_type"] if breaks else None
    broken_lvl = breaks[-1]["price"] if breaks else None

    # Fallback to spot boundaries if zones not yet formed
    if active_res_zone is None and spot > 0:
        active_res = round(spot * 1.008, 3)
        active_res_zone = SRZone(
            zone_type="RESISTANCE", high=active_res, low=round(spot * 1.004, 3),
            candle_open=spot, candle_close=active_res, timeframe=chosen_tf, pivot_bar_index=-1
        )
        all_levels.append(SRLevel(price=active_res, level_type="RESISTANCE", strength=3, source="ESTIMATED", timeframe=chosen_tf))

    if active_sup_zone is None and spot > 0:
        active_sup = round(spot * 0.992, 3)
        active_sup_zone = SRZone(
            zone_type="SUPPORT", high=round(spot * 0.996, 3), low=active_sup,
            candle_open=spot, candle_close=active_sup, timeframe=chosen_tf, pivot_bar_index=-1
        )
        all_levels.append(SRLevel(price=active_sup, level_type="SUPPORT", strength=3, source="ESTIMATED", timeframe=chosen_tf))

    # Kronos band levels
    if fc_lo is not None and fc_lo > 0:
        all_levels.append(SRLevel(price=fc_lo, level_type="SUPPORT", strength=4, source="KRONOS_BAND", timeframe="model"))
    if fc_hi is not None and fc_hi > 0:
        all_levels.append(SRLevel(price=fc_hi, level_type="RESISTANCE", strength=4, source="KRONOS_BAND", timeframe="model"))

    supports = [lvl for lvl in all_levels if lvl.price < spot]
    resistances = [lvl for lvl in all_levels if lvl.price > spot]

    nearest_sup = max(supports, key=lambda x: x.price).price if supports else active_sup
    nearest_res = min(resistances, key=lambda x: x.price).price if resistances else active_res

    dist_sup_pct = ((spot - nearest_sup) / spot * 100.0) if (nearest_sup and spot > 0) else 999.0
    dist_res_pct = ((nearest_res - spot) / spot * 100.0) if (nearest_res and spot > 0) else 999.0

    # 4. Multi-Timeframe Targets & Confluences (for cross-timeframe TP and SL support)
    higher_tf_targets = {
        "5m_res": by_timeframe.get("5m", {}).get("active_resistance"),
        "5m_sup": by_timeframe.get("5m", {}).get("active_support"),
        "15m_res": by_timeframe.get("15m", {}).get("active_resistance"),
        "15m_sup": by_timeframe.get("15m", {}).get("active_support"),
        "30m_res": by_timeframe.get("30m", {}).get("active_resistance"),
        "30m_sup": by_timeframe.get("30m", {}).get("active_support"),
        "1h_res": by_timeframe.get("1h", {}).get("active_resistance"),
        "1h_sup": by_timeframe.get("1h", {}).get("active_support"),
        "4h_res": by_timeframe.get("4h", {}).get("active_resistance"),
        "4h_sup": by_timeframe.get("4h", {}).get("active_support"),
    }

    # Consolidate unique levels
    unique_levels: List[SRLevel] = []
    seen_prices = []
    for lvl in sorted(all_levels, key=lambda x: (x.strength, -abs(spot - x.price)), reverse=True):
        if not any(abs(lvl.price - p) / p < 0.0008 for p in seen_prices):
            seen_prices.append(lvl.price)
            unique_levels.append(lvl)

    return SRContext(
        symbol=symbol,
        spot=spot,
        active_resistance=active_res,
        active_support=active_sup,
        nearest_support=nearest_sup,
        nearest_resistance=nearest_res,
        dist_to_support_pct=dist_sup_pct,
        dist_to_resistance_pct=dist_res_pct,
        zone=zone,
        volume_osc=volume_osc,
        latest_break=latest_break,
        broken_level=broken_lvl,
        all_levels=unique_levels[:10],
        break_events=all_break_events,
        active_res_zone=active_res_zone,
        active_sup_zone=active_sup_zone,
        all_zones=all_zones,
        model_zone_verdict=model_zone_verdict,
        model_zone_reason=model_zone_reason,
        entry_sr_timeframe=chosen_tf,
        balance_bracket=balance_bracket,
        by_timeframe=by_timeframe,
        higher_tf_targets=higher_tf_targets,
    )
