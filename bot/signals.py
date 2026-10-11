"""Deriv Multiplier Signal Generation & Confluence Engine for Gold & Silver.

High-Quality Setup & Challenge Filter Architecture:
1. 4H Macro Trend Confluence + 1H Intermediate Trend.
2. Dual Setup Archetypes:
   - S/R Clean Bounces (Reversal with Wick Rejection)
   - S/R Structural Breakouts (Momentum Expansion through Key Levels)
3. Strict Confluence Scoring (0-100) — Only Grade A+ (>=90) & Grade A (>=80) permitted.
4. Dynamic Multiplier Sizing (capital protected against liquidation).
5. High-Impact Macro News Breakout Window.
6. Gold/Silver Cross-Asset Confirmation & Session Overlap.
7. Secondary Validator AI Confirmation (Llama 3.2 3B).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import Forecast
from .news_engine import NewsState
from .session_clock import SessionInfo
from .sr_levels import SRContext

log = logging.getLogger(__name__)

MULTUP, MULTDOWN, FLAT = "MULTUP", "MULTDOWN", "FLAT"


@dataclass
class MultiplierTradePlan:
    symbol: str                   # 'frxXAUUSD' or 'frxXAGUSD'
    direction: str                # 'MULTUP' | 'MULTDOWN' | 'FLAT'
    timeframe: str                # '5m', '15m', '30m'
    spot: float
    stop_loss: float              # Price level of tight Stop Loss
    take_profit: float            # Price level of Take Profit
    stop_loss_usd: float          # Guaranteed max dollar risk
    take_profit_usd: float        # Expected dollar reward
    rrr: float                    # Risk-to-Reward Ratio
    multiplier: int               # Selected multiplier (e.g. 100, 200, 300, 500, 800)
    stake: float                  # Sized stake amount
    conviction: float             # Kronos entry probability
    macro_direction: str          # 1h macro direction
    macro_conviction: float
    macro_4h_direction: str       # 4h macro direction
    macro_4h_conviction: float
    sr_zone: str                  # 'AT_SUPPORT', 'AT_RESISTANCE', 'BREAKOUT_UP', 'BREAKOUT_DOWN'
    strategy_mode: str            # 'SR_BOUNCE', 'SR_BREAKOUT', 'NEWS_BREAKOUT'
    quality_grade: str            # 'A+', 'A', 'B', 'REJECTED'
    confluence_score: int = 0     # 0 to 100 score
    confluence_factors: List[str] = field(default_factory=list)
    model_floor: float = 0.0      # Model projected downside floor where price may fall
    model_reach: float = 0.0      # Model projected reach target
    reason: str = ""
    gates: Dict[str, Any] = field(default_factory=dict)
    validator: Optional[Dict[str, Any]] = None
    # Multi-Timeframe S/R Confluence Attributes
    entry_sr_timeframe: str = "15m"
    entry_sr_label: str = ""
    target_sr_timeframe: str = "1h"
    target_sr_label: str = ""
    sl_reasoning: str = ""
    tp_reasoning: str = ""

    @property
    def tradeable(self) -> bool:
        return self.direction in (MULTUP, MULTDOWN) and self.stake > 0 and self.stop_loss_usd > 0 and self.quality_grade in ("A+", "A")

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "timeframe": self.timeframe,
            "spot": round(self.spot, 3),
            "stop_loss": round(self.stop_loss, 3),
            "take_profit": round(self.take_profit, 3),
            "stop_loss_usd": round(self.stop_loss_usd, 2),
            "take_profit_usd": round(self.take_profit_usd, 2),
            "rrr": round(self.rrr, 2),
            "multiplier": self.multiplier,
            "stake": round(self.stake, 2),
            "conviction": round(self.conviction, 4),
            "macro_direction": self.macro_direction,
            "macro_conviction": round(self.macro_conviction, 4),
            "macro_4h_direction": self.macro_4h_direction,
            "macro_4h_conviction": round(self.macro_4h_conviction, 4),
            "sr_zone": self.sr_zone,
            "strategy_mode": self.strategy_mode,
            "quality_grade": self.quality_grade,
            "confluence_score": self.confluence_score,
            "confluence_factors": self.confluence_factors,
            "model_floor": round(self.model_floor, 3),
            "model_reach": round(self.model_reach, 3),
            "tradeable": bool(self.tradeable),
            "reason": str(self.reason),
            "gates": {
                k: {
                    **v,
                    "pass": bool(v.get("pass", False))
                } for k, v in self.gates.items()
            } if self.gates else {},
            "validator": self.validator,
            "entry_sr_timeframe": str(self.entry_sr_timeframe),
            "entry_sr_label": str(self.entry_sr_label),
            "target_sr_timeframe": str(self.target_sr_timeframe),
            "target_sr_label": str(self.target_sr_label),
            "sl_reasoning": str(self.sl_reasoning),
            "tp_reasoning": str(self.tp_reasoning),
        }


def extract_model_sl_tp(
    entry_fc: Optional[Forecast],
    spot: float,
    direction: str,
    sr_ctx: Optional[SRContext] = None,
    symbol: str = "frxXAUUSD",
    balance: Optional[float] = None,
) -> Tuple[float, float, float, float, str, str, str, str, str, str]:
    """Extracts dynamic Stop Loss and Take Profit levels directly from Kronos forecast trajectories,

    harmonized with multi-timeframe S/R boundaries according to account balance.
    Returns:
        (sl_price, tp_price, model_floor, model_reach, entry_sr_tf, entry_sr_label, target_sr_tf, target_sr_label, sl_reasoning, tp_reasoning)
    """
    buffer_pct = 0.0008 if symbol == "frxXAGUSD" else 0.0004
    min_dist_pct = 0.0005 if symbol == "frxXAGUSD" else 0.0003

    entry_sr_tf = sr_ctx.entry_sr_timeframe if sr_ctx else "15m"
    target_sr_tf = "1h"
    entry_sr_label = ""
    target_sr_label = ""
    sl_reasoning = ""
    tp_reasoning = ""

    # Default fallbacks if paths unavailable
    if direction == MULTUP:
        def_sl = round(spot * (1.0 - 0.0035), 3)
        def_tp = round(spot * (1.0 + 0.0070), 3)
    else:
        def_sl = round(spot * (1.0 + 0.0035), 3)
        def_tp = round(spot * (1.0 - 0.0070), 3)

    if entry_fc is None or not hasattr(entry_fc, "paths") or entry_fc.paths is None or len(entry_fc.paths) == 0:
        return def_sl, def_tp, def_sl, def_tp, entry_sr_tf, "Estimated S/R", target_sr_tf, "Estimated 1h Target", "Default fallback SL", "Default fallback TP"

    paths = entry_fc.paths
    path_lows = np.min(paths[:, :, 2], axis=1)
    path_highs = np.max(paths[:, :, 1], axis=1)

    max_sl_pct = 0.0075

    if direction == MULTUP:
        model_floor = float(np.percentile(path_lows, 15))
        model_reach = float(np.percentile(path_highs, 68))

        # Stop Loss: Anchored strictly to the entry timeframe's Support Zone Floor
        sl_candidate = model_floor * (1.0 - buffer_pct)
        if sr_ctx and sr_ctx.active_sup_zone and sr_ctx.active_sup_zone.low < spot:
            sl_candidate = min(sl_candidate, sr_ctx.active_sup_zone.low * (1.0 - buffer_pct))
            entry_sr_label = f"{entry_sr_tf} Support Zone [${sr_ctx.active_sup_zone.low:.2f} - ${sr_ctx.active_sup_zone.high:.2f}]"
            sl_reasoning = f"Tight SL (${sl_candidate:.2f}) below {entry_sr_tf} Support Floor for {sr_ctx.balance_bracket} capital preservation"
        elif sr_ctx and sr_ctx.nearest_support and sr_ctx.nearest_support < spot:
            sl_candidate = min(sl_candidate, sr_ctx.nearest_support * (1.0 - buffer_pct))
            entry_sr_label = f"Nearest Support (${sr_ctx.nearest_support:.2f})"
            sl_reasoning = f"Stop Loss below structural support (${sr_ctx.nearest_support:.2f})"
        else:
            entry_sr_label = f"Model Dip Floor (${model_floor:.2f})"
            sl_reasoning = f"Stop Loss at projected model floor (${model_floor:.2f})"

        # Multi-timeframe SL support check
        if sr_ctx and sr_ctx.higher_tf_targets:
            sup_1h = sr_ctx.higher_tf_targets.get("1h_sup")
            if sup_1h and sup_1h < spot and abs(sup_1h - sl_candidate) / spot < 0.0035:
                sl_reasoning += f" (reinforced by 1h Support Floor ${sup_1h:.2f})"

        sl_candidate = max(sl_candidate, spot * (1.0 - max_sl_pct))
        if sl_candidate >= spot * (1.0 - min_dist_pct):
            sl_candidate = spot * (1.0 - 0.0025)
        sl_price = round(sl_candidate, 3)

        # Take Profit: Cross-timeframe target mapping
        tp_candidate = model_reach
        target_sr_tf = "1h"
        target_sr_label = f"Model Reach (${model_reach:.2f})"
        tp_reasoning = f"Targeting model projected peak (${model_reach:.2f})"

        if sr_ctx and sr_ctx.higher_tf_targets:
            res_4h = sr_ctx.higher_tf_targets.get("4h_res")
            res_1h = sr_ctx.higher_tf_targets.get("1h_res")
            res_30m = sr_ctx.higher_tf_targets.get("30m_res")
            res_15m = sr_ctx.higher_tf_targets.get("15m_res")

            # Check higher timeframes in hierarchical order
            if res_1h and res_1h > spot and res_1h <= model_reach * 1.004:
                tp_candidate = max(tp_candidate, res_1h)
                target_sr_tf = "1h"
                target_sr_label = f"1h Resistance [${res_1h:.2f}]"
                tp_reasoning = f"Take Profit targets 1h Structural Resistance (${res_1h:.2f}) for high RRR run"
            elif res_30m and res_30m > spot and res_30m <= model_reach * 1.004:
                tp_candidate = max(tp_candidate, res_30m)
                target_sr_tf = "30m"
                target_sr_label = f"30m Resistance [${res_30m:.2f}]"
                tp_reasoning = f"Take Profit targets 30m Resistance (${res_30m:.2f})"
            elif res_15m and res_15m > spot:
                tp_candidate = max(tp_candidate, res_15m)
                target_sr_tf = "15m"
                target_sr_label = f"15m Resistance [${res_15m:.2f}]"
                tp_reasoning = f"Take Profit targets 15m Resistance (${res_15m:.2f})"

        if tp_candidate <= spot * (1.0 + min_dist_pct):
            tp_candidate = spot * (1.0 + 0.005)
        tp_price = round(tp_candidate, 3)

        floor_val = model_floor
        reach_val = model_reach

    elif direction == MULTDOWN:
        model_ceiling = float(np.percentile(path_highs, 85))
        model_reach = float(np.percentile(path_lows, 32))

        # Stop Loss: Anchored strictly to the entry timeframe's Resistance Zone Ceiling
        sl_candidate = model_ceiling * (1.0 + buffer_pct)
        if sr_ctx and sr_ctx.active_res_zone and sr_ctx.active_res_zone.high > spot:
            sl_candidate = max(sl_candidate, sr_ctx.active_res_zone.high * (1.0 + buffer_pct))
            entry_sr_label = f"{entry_sr_tf} Resistance Zone [${sr_ctx.active_res_zone.low:.2f} - ${sr_ctx.active_res_zone.high:.2f}]"
            sl_reasoning = f"Tight SL (${sl_candidate:.2f}) above {entry_sr_tf} Resistance Ceiling for {sr_ctx.balance_bracket} capital preservation"
        elif sr_ctx and sr_ctx.nearest_resistance and sr_ctx.nearest_resistance > spot:
            sl_candidate = max(sl_candidate, sr_ctx.nearest_resistance * (1.0 + buffer_pct))
            entry_sr_label = f"Nearest Resistance (${sr_ctx.nearest_resistance:.2f})"
            sl_reasoning = f"Stop Loss above structural resistance (${sr_ctx.nearest_resistance:.2f})"
        else:
            entry_sr_label = f"Model Ceiling (${model_ceiling:.2f})"
            sl_reasoning = f"Stop Loss at projected model ceiling (${model_ceiling:.2f})"

        # Multi-timeframe SL support check
        if sr_ctx and sr_ctx.higher_tf_targets:
            res_1h = sr_ctx.higher_tf_targets.get("1h_res")
            if res_1h and res_1h > spot and abs(res_1h - sl_candidate) / spot < 0.0035:
                sl_reasoning += f" (reinforced by 1h Resistance Ceiling ${res_1h:.2f})"

        sl_candidate = min(sl_candidate, spot * (1.0 + max_sl_pct))
        if sl_candidate <= spot * (1.0 + min_dist_pct):
            sl_candidate = spot * (1.0 + 0.0025)
        sl_price = round(sl_candidate, 3)

        # Take Profit: Cross-timeframe target mapping
        tp_candidate = model_reach
        target_sr_tf = "1h"
        target_sr_label = f"Model Downward Reach (${model_reach:.2f})"
        tp_reasoning = f"Targeting model projected downward reach (${model_reach:.2f})"

        if sr_ctx and sr_ctx.higher_tf_targets:
            sup_1h = sr_ctx.higher_tf_targets.get("1h_sup")
            sup_30m = sr_ctx.higher_tf_targets.get("30m_sup")
            sup_15m = sr_ctx.higher_tf_targets.get("15m_sup")

            if sup_1h and sup_1h < spot and sup_1h >= model_reach * 0.996:
                tp_candidate = min(tp_candidate, sup_1h)
                target_sr_tf = "1h"
                target_sr_label = f"1h Support [${sup_1h:.2f}]"
                tp_reasoning = f"Take Profit targets 1h Structural Support (${sup_1h:.2f}) for high RRR run"
            elif sup_30m and sup_30m < spot and sup_30m >= model_reach * 0.996:
                tp_candidate = min(tp_candidate, sup_30m)
                target_sr_tf = "30m"
                target_sr_label = f"30m Support [${sup_30m:.2f}]"
                tp_reasoning = f"Take Profit targets 30m Support (${sup_30m:.2f})"
            elif sup_15m and sup_15m < spot:
                tp_candidate = min(tp_candidate, sup_15m)
                target_sr_tf = "15m"
                target_sr_label = f"15m Support [${sup_15m:.2f}]"
                tp_reasoning = f"Take Profit targets 15m Support (${sup_15m:.2f})"

        if tp_candidate >= spot * (1.0 - min_dist_pct):
            tp_candidate = spot * (1.0 - 0.005)
        tp_price = round(tp_candidate, 3)

        floor_val = model_reach
        reach_val = model_ceiling

    else:
        sl_price = 0.0
        tp_price = 0.0
        floor_val = 0.0
        reach_val = 0.0
        entry_sr_label = "Standby"
        target_sr_label = "Standby"
        sl_reasoning = "None"
        tp_reasoning = "None"

    return sl_price, tp_price, round(floor_val, 3), round(reach_val, 3), entry_sr_tf, entry_sr_label, target_sr_tf, target_sr_label, sl_reasoning, tp_reasoning


def build_multiplier_plan(
    cfg,
    symbol: str,
    entry_fc: Forecast,
    macro_fc: Optional[Forecast],        # 1h macro forecast
    sr_ctx: SRContext,
    session: SessionInfo,
    news: NewsState,
    spot: float,
    balance: Optional[float] = None,
    cross_confirm: bool = False,
    validator_res: Optional[Any] = None,
    macro_4h_fc: Optional[Forecast] = None,
    candles_df: Optional[pd.DataFrame] = None,
) -> MultiplierTradePlan:
    """Builds an institutional-grade, high-confluence Multiplier plan for the $50-$100K challenge."""
    timeframe = cfg.get_active_timeframe(balance)
    up_prob = entry_fc.upside_prob
    down_prob = 1.0 - up_prob
    conviction = max(up_prob, down_prob)
    kronos_bias = MULTUP if up_prob > 0.5 else MULTDOWN if down_prob > 0.5 else FLAT

    # 1. 4H Macro Trend Evaluation (Primary Institutional Compass)
    if macro_4h_fc is not None:
        m4_up = macro_4h_fc.upside_prob
        m4_bias = MULTUP if m4_up > 0.52 else MULTDOWN if m4_up < 0.48 else FLAT
        m4_conv = max(m4_up, 1.0 - m4_up)
    else:
        m4_bias = "FLAT"
        m4_conv = 0.50

    # 2. 1H Macro Trend Evaluation (Intermediate Compass)
    if macro_fc is not None:
        m1_up = macro_fc.upside_prob
        m1_bias = MULTUP if m1_up > 0.52 else MULTDOWN if m1_up < 0.48 else FLAT
        m1_conv = max(m1_up, 1.0 - m1_up)
    else:
        m1_bias = "FLAT"
        m1_conv = 0.50

    # 3. Strategy Mode & Structural Level Identification
    direction = FLAT
    strategy_mode = "NONE"

    is_news_mode = news.allows_news_breakout and cfg.enable_news_trading

    # ---------------- Priority 1: High-Impact Macro News Breakout ----------------
    if is_news_mode and conviction >= 0.62:
        strategy_mode = "NEWS_BREAKOUT"
        direction = kronos_bias

    # ---------------- Priority 2: S/R Structural Breakouts ----------------
    elif (sr_ctx.zone == "BREAKOUT_UP" or sr_ctx.model_zone_verdict == "ZONE_BREAK_UP") and kronos_bias == MULTUP:
        strategy_mode = "SR_BREAKOUT"
        direction = MULTUP

    elif (sr_ctx.zone == "BREAKOUT_DOWN" or sr_ctx.model_zone_verdict == "ZONE_BREAK_DOWN") and kronos_bias == MULTDOWN:
        strategy_mode = "SR_BREAKOUT"
        direction = MULTDOWN

    # ---------------- Priority 3: S/R Clean Bounces (Reversal with Confirmation) ----------------
    elif kronos_bias == MULTUP and (sr_ctx.zone in ("AT_SUPPORT", "IN_SUPPORT_ZONE", "REVERSAL_UP") or sr_ctx.model_zone_verdict == "ZONE_REVERSAL_UP"):
        strategy_mode = "SR_BOUNCE"
        direction = MULTUP

    elif kronos_bias == MULTDOWN and (sr_ctx.zone in ("AT_RESISTANCE", "IN_RESISTANCE_ZONE", "REVERSAL_DOWN") or sr_ctx.model_zone_verdict == "ZONE_REVERSAL_DOWN"):
        strategy_mode = "SR_BOUNCE"
        direction = MULTDOWN

    # 4. Model-Driven Stop Loss & Take Profit Extraction (Multi-Timeframe Confluence)
    sl_price, tp_price, model_floor, model_reach, entry_sr_tf, entry_sr_label, target_sr_tf, target_sr_label, sl_reasoning, tp_reasoning = extract_model_sl_tp(
        entry_fc, spot, direction, sr_ctx, symbol, balance=balance
    )

    # 5. Natural Emergent Risk-to-Reward Ratio (RRR)
    sl_dist = abs(spot - sl_price) if (sl_price > 0 and direction != FLAT) else 999.0
    tp_dist = abs(tp_price - spot) if (tp_price > 0 and direction != FLAT) else 0.0
    rrr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0

    # 6. Dynamic Multiplier Sizing (Guaranteed capital preservation)
    multiplier, stake, target_loss_usd = cfg.calculate_multiplier_sizing(balance, spot, sl_price)
    expected_profit_usd = target_loss_usd * rrr

    # 6. Deep Confluence Scoring (0 - 100 Points)
    score = 0
    factors: List[str] = []

    if direction != FLAT:
        # Factor 1: 4H Macro Trend Confluence (up to 20 pts)
        if m4_bias == direction and m4_conv >= 0.54:
            score += 20
            factors.append(f"4H Macro Trend Aligned ({m4_bias} {m4_conv*100:.0f}%)")
        elif m4_bias == "FLAT":
            score += 10
            factors.append("4H Macro Neutral")
        elif m4_bias != direction and m4_conv >= 0.62:
            score -= 15
            factors.append(f"4H Counter-Trend Penalty ({m4_bias} {m4_conv*100:.0f}%)")

        # Factor 2: 1H Macro Alignment (up to 15 pts)
        if m1_bias == direction and m1_conv >= 0.54:
            score += 15
            factors.append(f"1H Intermediate Trend Aligned ({m1_bias} {m1_conv*100:.0f}%)")
        elif m1_bias == "FLAT":
            score += 8

        # Factor 3: Structural S/R Precision (up to 20 pts)
        if strategy_mode == "SR_BREAKOUT":
            score += 20
            factors.append(f"S/R Structural Breakout ({sr_ctx.zone})")
        elif strategy_mode == "SR_BOUNCE":
            score += 20
            factors.append(f"Clean S/R Reversal Zone ({sr_ctx.zone})")
        elif strategy_mode == "NEWS_BREAKOUT":
            score += 20
            factors.append("High-Impact Macro News Momentum Breakout")
        elif strategy_mode == "KRONOS_TREND":
            has_runway = (direction == MULTUP and sr_ctx.dist_to_resistance_pct >= 0.28) or \
                         (direction == MULTDOWN and sr_ctx.dist_to_support_pct >= 0.28)
            if has_runway:
                score += 20
                factors.append(f"Trend Momentum with Clear S/R Runway ({sr_ctx.zone})")
            else:
                score += 15
                factors.append(f"Trend Continuation Structure ({sr_ctx.zone})")

        # Factor 4: Candlestick Price Action Anatomy (up to 15 pts)
        if candles_df is not None and len(candles_df) >= 2:
            last_bar = candles_df.iloc[-1]
            b_high, b_low = float(last_bar["high"]), float(last_bar["low"])
            b_open, b_close = float(last_bar["open"]), float(last_bar["close"])
            bar_rng = max(0.0001, b_high - b_low)
            body = abs(b_close - b_open)
            l_wick = min(b_open, b_close) - b_low
            u_wick = b_high - max(b_open, b_close)

            if direction == MULTUP:
                if strategy_mode == "SR_BOUNCE" and (l_wick / bar_rng) >= 0.30:
                    score += 15
                    factors.append(f"Hammer Wick Rejection at Support ({(l_wick/bar_rng)*100:.0f}%)")
                elif strategy_mode == "SR_BREAKOUT" and b_close > b_open and (body / bar_rng) >= 0.45:
                    score += 15
                    factors.append("Bullish Breakout Expansion Candle")
                else:
                    score += 8
            elif direction == MULTDOWN:
                if strategy_mode == "SR_BOUNCE" and (u_wick / bar_rng) >= 0.30:
                    score += 15
                    factors.append(f"Shooting Star Rejection at Resistance ({(u_wick/bar_rng)*100:.0f}%)")
                elif strategy_mode == "SR_BREAKOUT" and b_close < b_open and (body / bar_rng) >= 0.45:
                    score += 15
                    factors.append("Bearish Breakdown Expansion Candle")
                else:
                    score += 8
        else:
            score += 10

        # Factor 5: Kronos Entry Conviction (up to 15 pts)
        if conviction >= 0.65:
            score += 15
            factors.append(f"High Kronos Foundation Conviction ({conviction*100:.1f}%)")
        elif conviction >= 0.60:
            score += 10
            factors.append(f"Kronos Conviction ({conviction*100:.1f}%)")

        # Factor 6: Session Golden Overlap & Liquidity (up to 10 pts)
        if session.is_golden_overlap:
            score += 10
            factors.append("London/NY Golden Overlap Liquidity Surge")
        elif session.liquidity_state in ("PEAK_LIQUIDITY", "NORMAL"):
            score += 6
            factors.append(f"Session Active ({session.liquidity_state})")

        # Factor 7: Cross-Metal Confirmation (up to 5 pts)
        if cross_confirm:
            score += 5
            factors.append("Gold & Silver Directional Harmony")

        # Factor 8: Secondary Validator AI Confirmation (Bonus 10 pts)
        v_dict = None
        if cfg.enable_validator_model and validator_res is not None:
            v_dict = validator_res.to_dict() if hasattr(validator_res, "to_dict") else validator_res
            if v_dict.get("verdict") == "CONFIRM":
                score += 10
                factors.append(f"Validator {v_dict.get('model', 'Llama 3.2')} Confirmed")
    else:
        v_dict = None

    score = max(0, min(100, score))

    # 7. Quality Grading & Strict Challenge Filter
    candidate_direction = direction
    if candidate_direction != FLAT:
        if score >= 90:
            grade = "A+"
        elif score >= 80:
            grade = "A"
        else:
            grade = "REJECTED"
            direction = FLAT
    else:
        grade = "REJECTED"

    # 8. Safety & Quality Gates (evaluated against candidate trade setup)
    gates: Dict[str, Any] = {}

    gates["setup_confluence"] = {
        "score": score,
        "min": 80,
        "pass": score >= 80 and grade in ("A+", "A"),
    }

    # Permanent 4H Macro Agreement Gate: always strictly enforced at all times
    macro_pass = (m4_bias == "FLAT") or (strategy_mode == "NEWS_BREAKOUT") or (m4_bias == candidate_direction) or (m4_conv < 0.58)

    gates["macro_4h"] = {
        "bias_4h": m4_bias,
        "conviction_4h": round(m4_conv, 3),
        "trade_direction": candidate_direction,
        "pass": macro_pass,
    }

    gates["conviction"] = {
        "value": round(conviction, 4),
        "min": cfg.min_prob,
        "pass": conviction >= cfg.min_prob and candidate_direction in (MULTUP, MULTDOWN),
    }

    dynamic_mode = getattr(cfg, "dynamic_model_tp_sl", True)
    min_req_rrr = 1.0 if dynamic_mode else cfg.min_rrr
    gates["rrr"] = {
        "value": round(rrr, 2),
        "min": min_req_rrr,
        "pass": rrr >= min_req_rrr if candidate_direction != FLAT else True,
    }

    session_pass = (
        (not cfg.enforce_session_filter)
        or (session.liquidity_state in ("PEAK_LIQUIDITY", "NORMAL"))
        or (strategy_mode == "NEWS_BREAKOUT")
        or (score >= 80 and grade in ("A+", "A"))
    )
    gates["session_liquidity"] = {
        "state": session.liquidity_state,
        "pass": session_pass,
    }

    if cfg.enable_validator_model and v_dict:
        v_verdict = v_dict.get("verdict", "CAUTION")
        v_conf = float(v_dict.get("confidence", 0.0))
        is_pass = (v_verdict != "REJECT") and (v_conf >= cfg.validator_min_confidence) if cfg.validator_require_confirmation else True
        gates["validator_ai"] = {
            "model": cfg.validator_model_name,
            "verdict": v_verdict,
            "confidence": v_conf,
            "pass": is_pass,
        }

    # Permanent S/R Zone Anchor Gate: always strictly enforced for all non-news trades
    if strategy_mode != "NEWS_BREAKOUT":
        sr_interacted = (sr_ctx.zone in ("IN_SUPPORT_ZONE", "IN_RESISTANCE_ZONE", "AT_SUPPORT", "AT_RESISTANCE", "REVERSAL_UP", "REVERSAL_DOWN", "BREAKOUT_UP", "BREAKOUT_DOWN")) or (sr_ctx.model_zone_verdict in ("ZONE_REVERSAL_UP", "ZONE_REVERSAL_DOWN", "ZONE_BREAK_UP", "ZONE_BREAK_DOWN"))
        gates["sr_zone_anchor"] = {
            "zone": sr_ctx.zone,
            "model_verdict": sr_ctx.model_zone_verdict,
            "pass": sr_interacted,
        }

    failed_gates = [k for k, v in gates.items() if not v["pass"]]

    if failed_gates and direction != FLAT:
        direction = FLAT
        grade = "REJECTED"

    plan = MultiplierTradePlan(
        symbol=symbol,
        direction=direction,
        timeframe=timeframe,
        spot=spot,
        stop_loss=sl_price,
        take_profit=tp_price,
        stop_loss_usd=target_loss_usd,
        take_profit_usd=expected_profit_usd,
        rrr=rrr,
        multiplier=multiplier,
        stake=stake,
        conviction=conviction,
        macro_direction=m1_bias,
        macro_conviction=m1_conv,
        macro_4h_direction=m4_bias,
        macro_4h_conviction=m4_conv,
        sr_zone=sr_ctx.zone,
        strategy_mode=strategy_mode,
        quality_grade=grade,
        confluence_score=score,
        confluence_factors=factors,
        model_floor=model_floor,
        model_reach=model_reach,
        gates=gates,
        validator=v_dict,
        entry_sr_timeframe=entry_sr_tf,
        entry_sr_label=entry_sr_label,
        target_sr_timeframe=target_sr_tf,
        target_sr_label=target_sr_label,
        sl_reasoning=sl_reasoning,
        tp_reasoning=tp_reasoning,
    )

    if failed_gates:
        plan.reason = f"Blocked: {', '.join(failed_gates)} (Score: {score}/100)"
    elif direction == FLAT:
        plan.reason = f"Monitoring {sr_ctx.entry_sr_timeframe} S/R ({sr_ctx.balance_bracket}) · Target S/R: {target_sr_tf}"
    else:
        plan.reason = (f"Grade {grade} [{score}/100] {direction} | {entry_sr_label} -> {target_sr_label} | "
                       f"{multiplier}x Stake ${stake:.2f} | RRR {rrr:.1f}:1")

    return plan
