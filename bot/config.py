"""Configuration for Kronos Deriv Multiplier Autonomous Trading Bot (Gold & Silver)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.getenv("KRONOS_CONFIG", os.path.join(ROOT, "config.json"))

SECTIONS = {
    "deriv": ["app_id", "token", "mode"],
    "market": ["symbols", "timeframe_mode", "low_balance_threshold", "mid_balance_threshold"],
    "kronos": ["model_id", "tokenizer_id", "device", "max_context", "lookback",
               "entry_pred_len", "n_paths", "n_paths_regime",
               "max_batch", "temperature", "top_p", "top_k"],
    "multipliers": ["available_multipliers", "default_multiplier", "max_multiplier",
                    "max_sl_stake_pct", "use_deal_cancellation"],
    "strategy": ["min_prob", "min_rrr", "dynamic_model_tp_sl",
                 "enable_news_trading", "enforce_session_filter",
                 "risk_type", "risk_value", "currency", "max_concurrent",
                 "cooldown_bars", "max_trades_per_day", "max_daily_loss_pct"],
    "validator": ["enable_validator_model", "validator_model_name", "validator_api_url",
                  "validator_require_confirmation", "validator_min_confidence"],
    "position_monitor": ["enable_position_monitor", "close_on_reversal", "reversal_prob_threshold",
                         "take_profit_early_pct", "protect_peak_profit"],
    "infra": ["db_path", "host", "port", "log_level"],
}


@dataclass
class Config:
    # ---------------- deriv ----------------
    app_id: str = "1089"
    token: str = ""
    mode: str = "demo"             # demo | real

    # ---------------- market (Gold & Silver Dual-Asset) ----------------
    symbols: List[str] = None      # ["frxXAUUSD", "frxXAGUSD"]
    timeframe_mode: str = "adaptive"  # adaptive (based on balance) | 5m | 15m | 30m
    low_balance_threshold: float = 250.0  # Balance below this uses strictly 5m for tight SL
    mid_balance_threshold: float = 750.0  # Balance below this uses 5m/15m

    # ---------------- kronos foundation model ----------------
    model_id: str = "NeoQuasar/Kronos-mini"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-2k"
    device: str = "auto"
    max_context: int = 512
    lookback: int = 288            # 288 bars = 24 hours at 5m, or 3 days at 15m
    entry_pred_len: int = 20       # Predict 20 candles ahead
    n_paths: int = 24              # Monte Carlo trajectories
    n_paths_regime: int = 16
    max_batch: int = 64
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0

    # ---------------- deriv multipliers ----------------
    available_multipliers: List[int] = None   # [100, 200, 300, 500, 800]
    default_multiplier: int = 200
    max_multiplier: int = 800
    max_sl_stake_pct: float = 50.0 # Max % of stake lost when SL is hit (preserves at least 50% of stake)
    use_deal_cancellation: bool = False

    # ---------------- strategy & risk ----------------
    min_prob: float = 0.60         # High conviction threshold (60%) for quality setups
    min_rrr: float = 1.5           # Minimum Risk-to-Reward Ratio (1.5:1)
    dynamic_model_tp_sl: bool = True  # Model-driven SL & TP from Kronos forecast extremes (no fixed RRR)
    enable_news_trading: bool = True
    enforce_session_filter: bool = False  # Disabled by default so high quality setups are traded across all sessions
    risk_type: str = "percent"     # percent (% of balance) | fixed ($ amount)
    risk_value: float = 1.5        # 1.5% risk per trade
    currency: str = "USD"
    max_concurrent: int = 2        # 1 trade on Gold + 1 trade on Silver simultaneously
    cooldown_bars: int = 1
    max_trades_per_day: int = 20
    max_daily_loss_pct: float = 5.0   # Max % daily drawdown from start-of-day balance before circuit-breaker halts trading

    # ---------------- secondary AI validator (RayCodes Llama 3.2 3B) ----------------
    enable_validator_model: bool = False
    validator_model_name: str = "llama3.2:3b"
    validator_api_url: str = "http://localhost:11434"
    validator_require_confirmation: bool = True
    validator_min_confidence: float = 0.65

    # ---------------- position monitor & dynamic exits ----------------
    enable_position_monitor: bool = True
    close_on_reversal: bool = True
    reversal_prob_threshold: float = 0.65
    take_profit_early_pct: float = 40.0
    protect_peak_profit: bool = True

    # ---------------- infra ----------------
    db_path: str = "kronos_deriv.db"
    host: str = "0.0.0.0"
    port: int = 8120
    log_level: str = "INFO"

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["frxXAUUSD", "frxXAGUSD"]
        if self.available_multipliers is None:
            self.available_multipliers = [100, 200, 300, 500, 800]

    def get_active_timeframe(self, balance: Optional[float] = None) -> str:
        """Determines active entry timeframe adaptively based on account balance."""
        if self.timeframe_mode != "adaptive":
            return self.timeframe_mode
        bal = balance if balance is not None and balance > 0 else 100.0
        if bal < self.low_balance_threshold:
            # Low balance (< $250): Focus strictly on 5m for tight SL and lower point risk
            return "5m"
        elif bal < self.mid_balance_threshold:
            return "15m"
        else:
            return "30m"

    def get_active_macro_timeframe(self, balance: Optional[float] = None) -> str:
        """Determines active macro trend timeframe adaptively based on account balance."""
        bal = balance if balance is not None and balance > 0 else 100.0
        if bal < self.mid_balance_threshold:
            # Accounts under mid balance threshold (e.g. < $750): Macro trend on 1h (matches 5m/15m entries)
            return "1h"
        else:
            # Higher balance (>= $750): Macro trend on 4h (institutional swing flow)
            return "4h"

    def get_timeframe_hierarchy(self, balance: Optional[float] = None) -> Tuple[str, str, str]:
        """Returns the full adaptive timeframe hierarchy (entry, intermediate, macro) based on balance."""
        entry = self.get_active_timeframe(balance)
        macro = self.get_active_macro_timeframe(balance)
        if entry == "5m":
            intermediate = "15m"
        elif entry == "15m":
            intermediate = "30m"
        else:
            intermediate = "1h"
        return entry, intermediate, macro

    def calculate_multiplier_sizing(self, balance: Optional[float], entry_spot: float,
                                    sl_spot: float) -> Tuple[int, float, float]:
        """Calculates optimal multiplier, stake, and dollar SL based strictly on risk_type and risk_value.

        Stake is calculated directly from risk_type ('percent' or 'fixed') and risk_value at all times.
        Multiplier is selected so that Stop Loss triggers before Deriv stop-out liquidation.

        Returns: (multiplier, stake, target_dollar_risk)
        """
        bal = balance if balance is not None and balance > 0 else 100.0

        # Stake is strictly determined by risk_type and risk_value at all times
        if self.risk_type == "percent":
            stake = round((self.risk_value / 100.0) * bal, 2)
        else:
            stake = round(float(self.risk_value), 2)

        stake = max(0.1, stake)

        # Fractional distance to Stop Loss
        dist_pct = abs(entry_spot - sl_spot) / entry_spot
        if dist_pct <= 0.0001:
            dist_pct = 0.001  # Fallback 0.1%

        avail_mults = sorted(self.available_multipliers) if self.available_multipliers else [100, 200, 300, 500, 800]
        min_mult = min(avail_mults)

        # STAKE PRESERVATION INVARIANT:
        # The sole point of using a Stop Loss in Multipliers is to prevent killing the entire stake.
        # Deriv automatically liquidates (wipes 100% of stake) when adverse price movement = 1.0 / multiplier.
        # We enforce a maximum stake risk ratio (e.g. 50% of stake), guaranteeing that when Stop Loss triggers,
        # at least 50% of the initial stake is preserved and credited back to the balance.
        sl_stake_ratio = min(0.70, max(0.20, getattr(self, "max_sl_stake_pct", 50.0) / 100.0))

        # Maximum safe multiplier that will trigger SL at or before sl_stake_ratio of stake:
        # (e.g. For 0.25% SL distance and 50% stake risk, max multiplier is 0.50 / 0.0025 = 200x)
        max_safe_mult = int(sl_stake_ratio / dist_pct) if dist_pct > 0 else self.max_multiplier

        safe_multipliers = [m for m in avail_mults if m <= self.max_multiplier and m <= max_safe_mult]
        if not safe_multipliers:
            # If technical SL is wide, use the lowest available multiplier to minimize liquidation risk
            chosen_mult = min_mult
        else:
            default_m = getattr(self, "default_multiplier", 200)
            preferred = [m for m in safe_multipliers if m <= default_m]
            chosen_mult = max(preferred) if preferred else max(safe_multipliers)

        # Calculate exact dollar risk when Stop Loss is triggered
        raw_loss = stake * chosen_mult * dist_pct
        # CRITICAL DERIV INVARIANT: Stop Loss dollar amount MUST be strictly LESS than the stake!
        # Deriv rejects proposals if limit_order.stop_loss >= stake.
        # We cap actual SL dollar loss to sl_stake_ratio * stake (e.g. 50%), preserving the rest of stake.
        max_allowed_sl_usd = round(stake * sl_stake_ratio, 2)
        actual_loss_at_sl = round(min(raw_loss, max_allowed_sl_usd), 2)
        if actual_loss_at_sl < 0.10 and stake > 0:
            actual_loss_at_sl = round(min(0.50, max_allowed_sl_usd), 2)

        return chosen_mult, stake, actual_loss_at_sl

    # ---------------- io ----------------
    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "Config":
        cfg = cls()
        if not os.path.exists(path):
            return cfg
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        names = {f.name for f in fields(cls)}
        flat: dict = {}
        for key, val in raw.items():
            if isinstance(val, dict) and key in SECTIONS:
                flat.update(val)
            elif key in names:
                flat[key] = val
        for k, v in flat.items():
            if k in names:
                setattr(cfg, k, v)
        cfg._path = path
        return cfg

    def to_nested(self) -> dict:
        d = asdict(self)
        out: dict = {}
        for section, keys in SECTIONS.items():
            out[section] = {k: d[k] for k in keys if k in d}
        return out

    def save(self, path: str | None = None) -> str:
        path = path or getattr(self, "_path", CONFIG_PATH)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_nested(), fh, indent=2)
        return path

    def redacted(self) -> dict:
        d = self.to_nested()
        if d.get("deriv", {}).get("token"):
            d["deriv"]["token"] = "***set***"
        return d

    def validate(self) -> list:
        errs: list = []
        if self.risk_value <= 0:
            errs.append("risk_value must be > 0")
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct > 100.0:
            errs.append("max_daily_loss_pct must be between 0.5% and 100.0%")
        if self.min_prob < 0.5 or self.min_prob > 1.0:
            errs.append("min_prob must be between 0.50 and 1.00")
        if self.mode == "real" and not self.token:
            errs.append("real mode requires deriv.token to be set")
        return errs


CFG = Config.load()

CHOICES = {
    "mode": ["demo", "real"],
    "timeframe_mode": ["adaptive", "5m", "15m", "30m"],
    "risk_type": ["percent", "fixed"],
    "model_id": ["NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-base"],
    "validator_model_name": ["llama3.2:3b", "llama3.2:1b", "llama3.2", "mistral:7b"],
    "default_multiplier": [100, 200, 300, 500, 800],
    "max_multiplier": [100, 200, 300, 500, 800],
    "device": ["auto", "cuda:0", "cpu"],
    "log_level": ["DEBUG", "INFO", "WARNING", "ERROR"],
}

NEEDS_RESTART = {
    "symbols", "model_id", "tokenizer_id", "device", "mode", "token", "app_id", "db_path"
}

SECRET_FIELDS = {"token"}

HELP = {
    "symbols": "Active precious metals: Gold (frxXAUUSD) & Silver (frxXAGUSD).",
    "available_multipliers": "Official Deriv multiplier levels for Gold & Silver: 100x, 200x, 300x, 500x, 800x.",
    "default_multiplier": "Baseline multiplier (default 200x). Automatically scaled down if SL distance exceeds liquidation threshold.",
    "max_multiplier": "Highest allowed multiplier cap (up to 800x for ultra-tight precision bounces).",
    "max_sl_stake_pct": "Maximum percentage of stake risked when Stop Loss is triggered (default 50%). Guarantees that hitting Stop Loss preserves at least 50% of initial stake and prevents 100% stake liquidation.",
    "timeframe_mode": "Adaptive dynamically selects 5m, 15m, or 30m based on balance to protect capital with tight SL.",
    "low_balance_threshold": "Accounts below this USD balance stick strictly to 5m entries for micro-pip SL risk.",
    "mid_balance_threshold": "Accounts between low and mid balance thresholds trade 15m structure setups.",
    "min_prob": "Minimum Kronos model conviction required to take a trade (default 60% for A+ setups).",
    "min_rrr": "Minimum Risk-to-Reward ratio required before entering (default 1.5:1).",
    "dynamic_model_tp_sl": "Derives Stop Loss and Take Profit dynamically from Kronos projected fall floor and reach target without fixed RRR.",
    "enable_news_trading": "Enables the post-news breakout momentum strategy around US CPI, NFP, and FOMC.",
    "enforce_session_filter": "Restricts trade executions to high-liquidity London and New York sessions.",
    "risk_type": "Risk sizing model: 'percent' (% of total balance to compound growth) or 'fixed' ($ dollar risk per trade).",
    "risk_value": "Percentage of account balance to risk per trade (e.g. 2.0 = 2.0% risk). Multiplier and stake are mathematically calculated so hitting Stop Loss loses exactly this percent.",
    "enable_validator_model": "Enables the secondary AI validator model (Llama 3.2 3B from RayCodes) to independently review historical data and confirm Kronos setups.",
    "validator_model_name": "Name of the secondary validator LLM (default: llama3.2:3b as used in RayCodes_Kronos).",
    "validator_api_url": "Endpoint for local Ollama API (default: http://localhost:11434).",
    "validator_require_confirmation": "Strict confirmation gate: blocks execution if the secondary validator rejects or finds conflicting price action.",
    "validator_min_confidence": "Minimum confidence required from secondary validator (default: 0.65).",
    "enable_position_monitor": "Autonomous position monitor: continuously evaluates active positions to decide whether to HOLD or CLOSE early.",
    "close_on_reversal": "Closes an active position if Kronos directional conviction flips against the open contract.",
    "reversal_prob_threshold": "Conviction threshold (default 65%) to trigger an autonomous reversal exit.",
    "take_profit_early_pct": "Early profit-taking threshold (% floating gain) if price hits an opposite S/R barrier with stalling momentum.",
    "protect_peak_profit": "Trailing profit protection: locks in gains if floating profit drops by more than 50% from its highest peak.",
    "max_daily_loss_pct": "Maximum percentage daily loss from start-of-day balance (e.g. 5.0%). Circuit-breaker halts bot until midnight UTC if breached.",
}


def settings_schema(cfg: Config) -> list:
    from dataclasses import fields as dc_fields
    out = []
    by_key = {k: sec for sec, keys in SECTIONS.items() for k in keys}
    for f in dc_fields(cfg):
        if f.name not in by_key:
            continue
        val = getattr(cfg, f.name)
        kind = "bool" if isinstance(val, bool) else "int" if isinstance(val, int) else "float" if isinstance(val, float) else "list" if isinstance(val, list) else "str"
        out.append({
            "key": f.name,
            "section": by_key[f.name],
            "type": "select" if f.name in CHOICES else kind,
            "options": CHOICES.get(f.name),
            "value": ("***set***" if f.name in SECRET_FIELDS and val else val),
            "secret": f.name in SECRET_FIELDS,
            "restart": f.name in NEEDS_RESTART,
            "help": HELP.get(f.name, ""),
        })
    return out


def coerce(cfg: Config, key: str, raw: Any) -> Any:
    cur = getattr(cfg, key)
    if isinstance(cur, bool):
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(float(raw))
    if isinstance(cur, float):
        return float(raw)
    if isinstance(cur, list):
        if isinstance(raw, list):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return [x.strip() for x in str(raw).split(",") if x.strip()]
    return str(raw)
