"""Autonomous Position Monitor for Active Deriv Multiplier Contracts.

Continuously monitors open multiplier positions and uses Kronos foundation model
forecasts, algorithmic Support & Resistance levels, trailing profit dynamics,
and macro news alerts to determine whether the model should HOLD or CLOSE
an active position before the hard Stop Loss or Take Profit is hit.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HOLD = "HOLD"
CLOSE = "CLOSE"


class PositionMonitor:
    """Evaluates active multiplier positions and determines optimal holding or closing actions."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._reversal_warnings: Dict[int, int] = {}  # contract_id -> consecutive reversal evaluation ticks

    def evaluate_position(
        self,
        contract: Any,
        current_spot: float,
        latest_forecast: Optional[Any],
        sr_ctx: Optional[Any],
        news: Optional[Any],
        candles_df: Optional[pd.DataFrame] = None,
        macro_fc: Optional[Any] = None,
    ) -> Tuple[str, str]:
        """Evaluates whether to HOLD or CLOSE an active multiplier contract.

        Core Invariant:
        - Let winners run: Never close a trade with positive floating P&L on lower-timeframe micro-fluctuations.
        - Reversal cuts apply strictly during drawdown when confirmed by persistence (>= 2 ticks) and not blocked by macro trend.
        - Hard Stop Loss and Take Profit remain the primary institutional boundaries.

        Returns: (action: 'HOLD' | 'CLOSE', reason: str)
        """
        if not contract or contract.status != "open":
            return HOLD, "Contract not open"

        cid = getattr(contract, "contract_id", 0)

        # Update highest floating profit percentage reached
        cur_pnl_pct = float(getattr(contract, "profit_percentage", 0.0) or 0.0)
        highest_pnl = float(getattr(contract, "highest_profit_pct", 0.0) or 0.0)
        if cur_pnl_pct > highest_pnl:
            contract.highest_profit_pct = cur_pnl_pct
            highest_pnl = cur_pnl_pct

        direction = contract.direction  # 'MULTUP' or 'MULTDOWN'
        multiplier = contract.multiplier
        entry_spot = contract.entry_spot
        profit = float(getattr(contract, "profit", 0.0) or 0.0)
        elapsed_sec = time.time() - getattr(contract, "opened_ts", time.time())

        # Give the contract a grace period (25 seconds) after entry so spread and initial bar settle
        if elapsed_sec < 25.0:
            return HOLD, f"Entry grace period ({int(elapsed_sec)}s/25s) · Structure intact"

        # ---------------- 1. Emergency Pre-News Risk Cut ----------------
        if news and getattr(self.cfg, "enable_news_trading", True):
            mins_until = getattr(news, "minutes_until_event", None)
            mode = getattr(news, "mode", "NORMAL")
            # If high-impact news (CPI, NFP, FOMC) is within 2 minutes and position is NOT in news mode
            if mins_until is not None and 0 <= mins_until <= 2 and mode == "PRE_NEWS_ALERT":
                ev_name = getattr(news.active_event, "name", "Macro News") if news.active_event else "Macro News"
                return CLOSE, f"Emergency Macro News Exit: High-impact {ev_name} in {mins_until:.1f}m. Closing to avoid volatility spike."

        # ---------------- 2. Trailing Peak Profit Protection ----------------
        # If profit reached >= +35% and has retreated significantly (protecting windfall gains)
        protect_peak = getattr(self.cfg, "protect_peak_profit", True)
        if protect_peak and highest_pnl >= 35.0:
            drawdown_from_peak = highest_pnl - cur_pnl_pct
            if drawdown_from_peak >= (highest_pnl * 0.5) and cur_pnl_pct < (highest_pnl * 0.5):
                return CLOSE, f"Trailing Profit Lock: Profit retreated from peak +{highest_pnl:.1f}% to +{cur_pnl_pct:.1f}%. Locking in +${profit:.2f} gain."

        # ---------------- 3. S/R Target Reached & Momentum Stall ----------------
        # When floating profit is healthy (>= 40%) and price hits opposite S/R barrier with stalling price action
        early_tp_pct = getattr(self.cfg, "take_profit_early_pct", 40.0)
        if sr_ctx and cur_pnl_pct >= early_tp_pct:
            zone = getattr(sr_ctx, "zone", "MID_RANGE")
            if direction == "MULTUP" and zone in ("AT_RESISTANCE", "BREAKOUT_DOWN"):
                res_price = float(getattr(sr_ctx, "nearest_resistance", 0.0) or 0.0)
                return CLOSE, f"Target S/R Reached: Hit resistance at ${res_price:.2f} with +{cur_pnl_pct:.1f}% profit (+${profit:.2f}). Booking profit."

            elif direction == "MULTDOWN" and zone in ("AT_SUPPORT", "BREAKOUT_UP"):
                sup_price = float(getattr(sr_ctx, "nearest_support", 0.0) or 0.0)
                return CLOSE, f"Target S/R Reached: Hit support at ${sup_price:.2f} with +{cur_pnl_pct:.1f}% profit (+${profit:.2f}). Booking profit."

        # ---------------- 4. Kronos Reversal Detection (Hold Winners Principle) ----------------
        # INVARIANT: If the trade is in positive profit (cur_pnl_pct >= 0), DO NOT kill the trade early!
        # Multipliers yield asymmetric profit by letting winners run to TP. Lower timeframe pullbacks are normal.
        close_on_reversal = getattr(self.cfg, "close_on_reversal", True)
        rev_threshold = getattr(self.cfg, "reversal_prob_threshold", 0.70)

        if close_on_reversal and latest_forecast is not None:
            up_prob = getattr(latest_forecast, "upside_prob", 0.5)
            down_prob = 1.0 - up_prob

            is_reversal = (direction == "MULTUP" and down_prob >= rev_threshold) or \
                          (direction == "MULTDOWN" and up_prob >= rev_threshold)

            if is_reversal:
                # If trade is in profit, protect the run and HOLD
                if cur_pnl_pct >= 0.0:
                    self._reversal_warnings[cid] = 0
                    log.debug("Position Monitor: Reversal signal noted for #%s, but holding winner running in profit (+%.2f%%)", cid, cur_pnl_pct)
                else:
                    # In drawdown (cur_pnl_pct < 0): Check if macro trend still supports our position
                    macro_supports = False
                    if macro_fc is not None:
                        m_up = getattr(macro_fc, "upside_prob", 0.5)
                        if direction == "MULTUP" and m_up >= 0.52:
                            macro_supports = True
                        elif direction == "MULTDOWN" and (1.0 - m_up) >= 0.52:
                            macro_supports = True

                    if macro_supports:
                        self._reversal_warnings[cid] = 0
                        log.debug("Position Monitor: Macro trend aligns with #%s. Absorbing lower-TF counter-pullback.", cid)
                    else:
                        # Require persistent confirmation across at least 2 consecutive evaluation cycles
                        warn_count = self._reversal_warnings.get(cid, 0) + 1
                        self._reversal_warnings[cid] = warn_count
                        if warn_count >= 2:
                            rev_pct = (down_prob if direction == "MULTUP" else up_prob) * 100.0
                            opp_dir = "Bearish" if direction == "MULTUP" else "Bullish"
                            return CLOSE, f"Kronos Reversal Cut: Confirmed {rev_pct:.1f}% {opp_dir} against {direction} over 2+ cycles (P&L {cur_pnl_pct:.1f}%). Early risk cut."
            else:
                self._reversal_warnings[cid] = 0

        # ---------------- 5. Structure Breakdown Exit ----------------
        # If candles show clear momentum breakdown with negative P&L
        if candles_df is not None and len(candles_df) >= 3:
            recent = candles_df.iloc[-3:]
            closes = recent["close"].to_numpy()
            opens = recent["open"].to_numpy()

            if direction == "MULTUP" and cur_pnl_pct < -25.0:
                if np.all(closes < opens) and (closes[-1] < closes[0]):
                    return CLOSE, f"Structure Breakdown: 3 consecutive bear bars pushing against MULTUP. Early risk cut to prevent full SL hit."

            elif direction == "MULTDOWN" and cur_pnl_pct < -25.0:
                if np.all(closes > opens) and (closes[-1] > closes[0]):
                    return CLOSE, f"Structure Breakdown: 3 consecutive bull bars pushing against MULTDOWN. Early risk cut to prevent full SL hit."

        # ---------------- 5b. Emergency Stake Preservation Shield ----------------
        max_stake_loss_pct = float(getattr(self.cfg, "max_sl_stake_pct", 50.0) or 50.0)
        shield_threshold = -1.0 * min(max_stake_loss_pct * 0.90, 50.0)
        if cur_pnl_pct <= shield_threshold:
            return CLOSE, f"Emergency Stake Shield: Floating drawdown reached {cur_pnl_pct:.1f}%. Exiting early to preserve remaining stake."

        # ---------------- 6. Maintain Position (HOLD) ----------------
        status_detail = f"Floating: {cur_pnl_pct:+.1f}% (+${profit:.2f}) · Peak: +{highest_pnl:.1f}%"
        if latest_forecast:
            k_conf = getattr(latest_forecast, "upside_prob", 0.5)
            k_text = f"Kronos {k_conf*100:.1f}% UP" if direction == "MULTUP" else f"Kronos {(1.0-k_conf)*100:.1f}% DOWN"
            return HOLD, f"{status_detail} · {k_text} continuation supported"

        return HOLD, f"{status_detail} · Trade thesis intact"
