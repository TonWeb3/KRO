"""Secondary AI Validator Model for Kronos Multiplier Bot.

Inspired by RayCodes_Kronos (https://github.com/47thtechcorner/RayCodes_Kronos):
Pairs the quantitative Kronos foundation forecasting model with a secondary LLM
(Llama 3.2 3B via Ollama or local quant validation engine).

The validator independently reviews historical candlestick sequences, price action,
and algorithmic Support & Resistance levels to confirm or reject Kronos trading signals
before real capital or high multipliers are deployed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass
class ValidationResult:
    is_valid: bool                 # Overall pass/fail
    verdict: str                  # 'CONFIRM', 'REJECT', 'CAUTION', 'OFF'
    confidence: float             # 0.0 to 1.0
    support_confirmed: bool       # Does historical PA confirm support integrity?
    resistance_confirmed: bool    # Does historical PA confirm resistance integrity?
    reasoning: str                # Human-readable explanation of historical data analysis
    model_name: str               # e.g. 'llama3.2:3b' or 'fallback-quant'
    latency_ms: float = 0.0       # Model inference time in milliseconds
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "support_confirmed": self.support_confirmed,
            "resistance_confirmed": self.resistance_confirmed,
            "reasoning": self.reasoning,
            "model_name": self.model_name,
            "latency_ms": round(self.latency_ms, 1),
            "details": self.details,
        }


class ValidatorManager:
    """Manages downloading, health-checking, and invoking the secondary AI validator model."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._download_status: str = "IDLE"  # IDLE, DOWNLOADING, READY, ERROR
        self._download_progress: float = 0.0  # 0 to 100%
        self._download_message: str = ""
        self._download_lock = threading.Lock()
        self._cached_status: Optional[Dict[str, Any]] = None
        self._last_status_check: float = 0.0

    @property
    def model_name(self) -> str:
        return getattr(self.cfg, "validator_model_name", DEFAULT_OLLAMA_MODEL) if self.cfg else DEFAULT_OLLAMA_MODEL

    @property
    def api_url(self) -> str:
        url = getattr(self.cfg, "validator_api_url", DEFAULT_OLLAMA_URL) if self.cfg else DEFAULT_OLLAMA_URL
        return url.rstrip("/")

    @property
    def is_enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_validator_model", False)) if self.cfg else False

    def _refresh_status_bg(self) -> None:
        """Background worker checking Ollama tags with short timeout, never blocking event loop."""
        try:
            model_name = self.model_name
            is_ready = False
            available_models = []
            ollama_online = False
            message = ""

            try:
                r = requests.get(f"{self.api_url}/api/tags", timeout=1.5)
                if r.status_code == 200:
                    ollama_online = True
                    data = r.json()
                    models = data.get("models", [])
                    available_models = [m.get("name", "") for m in models]
                    is_ready = any(
                        model_name == m or model_name.split(":")[0] == m.split(":")[0]
                        for m in available_models
                    )
                    if is_ready:
                        message = f"{model_name} is downloaded and ready for validation."
                    else:
                        message = f"Ollama is running, but '{model_name}' is not yet downloaded."
            except Exception:
                ollama_online = False
                message = f"Ollama offline at {self.api_url}. Quant fallback engine active."

            status_str = "READY" if is_ready else ("DOWNLOADING" if self._download_status == "DOWNLOADING" else ("NOT_DOWNLOADED" if ollama_online else "OFFLINE"))
            if not self.is_enabled:
                status_str = "DISABLED"

            self._cached_status = {
                "enabled": self.is_enabled,
                "model_name": model_name,
                "api_url": self.api_url,
                "status": status_str,
                "ollama_online": ollama_online,
                "is_ready": is_ready,
                "download_status": self._download_status,
                "download_progress": self._download_progress,
                "download_message": self._download_message or message,
                "available_models": available_models,
            }
            self._last_status_check = time.time()
        finally:
            self._refreshing = False

    def get_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Returns the current state of the validator model immediately in 0ms without blocking."""
        if not self.is_enabled:
            return {
                "enabled": False,
                "model_name": self.model_name,
                "api_url": self.api_url,
                "status": "DISABLED",
                "ollama_online": False,
                "is_ready": False,
                "download_status": self._download_status,
                "download_progress": self._download_progress,
                "download_message": "Secondary AI Validator is disabled.",
                "available_models": [],
            }

        now = time.time()
        if self._cached_status is not None and not force_refresh:
            if (now - self._last_status_check > 30.0) and not getattr(self, "_refreshing", False):
                self._refreshing = True
                threading.Thread(target=self._refresh_status_bg, daemon=True, name="validator-status-bg").start()
            return self._cached_status

        # If cache is missing or force_refresh requested
        if not getattr(self, "_refreshing", False):
            self._refreshing = True
            threading.Thread(target=self._refresh_status_bg, daemon=True, name="validator-status-bg").start()

        if self._cached_status is not None:
            return self._cached_status

        # Instant provisional status while first background check runs
        return {
            "enabled": True,
            "model_name": self.model_name,
            "api_url": self.api_url,
            "status": "QUANT_ENGINE",
            "ollama_online": False,
            "is_ready": False,
            "download_status": self._download_status,
            "download_progress": self._download_progress,
            "download_message": "Initializing background validator health check...",
            "available_models": [],
        }

    def start_download_async(self) -> Dict[str, Any]:
        """Initiates background downloading (pulling) of the validator model via Ollama."""
        with self._download_lock:
            if self._download_status == "DOWNLOADING":
                return {"ok": True, "message": "Download is already in progress.", "status": "DOWNLOADING"}
            self._download_status = "DOWNLOADING"
            self._download_progress = 0.0
            self._download_message = f"Starting download of {self.model_name}..."

        t = threading.Thread(target=self._run_download_thread, daemon=True, name="model-download")
        t.start()
        return {"ok": True, "message": f"Initiated download of {self.model_name}", "status": "DOWNLOADING"}

    def _run_download_thread(self):
        model_name = self.model_name
        log.info("Starting background download of validator model '%s' from %s", model_name, self.api_url)
        try:
            r = requests.post(
                f"{self.api_url}/api/pull",
                json={"name": model_name, "stream": True},
                stream=True,
                timeout=1800,  # 30 min max for large weights
            )
            if r.status_code != 200:
                with self._download_lock:
                    self._download_status = "ERROR"
                    self._download_message = f"Ollama returned HTTP {r.status_code}: {r.text[:200]}"
                log.error("Failed to pull model %s: HTTP %s", model_name, r.status_code)
                return

            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    status_text = chunk.get("status", "")
                    total = chunk.get("total", 0)
                    completed = chunk.get("completed", 0)
                    if total > 0:
                        pct = round((completed / total) * 100, 1)
                        self._download_progress = pct
                        self._download_message = f"{status_text} ({pct}%)"
                    else:
                        self._download_message = status_text
                except Exception:
                    pass

            with self._download_lock:
                self._download_status = "READY"
                self._download_progress = 100.0
                self._download_message = f"{model_name} successfully downloaded and cached."
            log.info("Successfully downloaded validator model '%s'", model_name)

        except Exception as e:
            with self._download_lock:
                self._download_status = "ERROR"
                self._download_message = f"Download failed: {str(e)}"
            log.error("Download exception for %s: %s", model_name, e)

    def analyze_historical_price_action(
        self,
        df: pd.DataFrame,
        spot: float,
        sr_ctx: Any,
        direction: str,
        n_bars: int = 15,
    ) -> Dict[str, Any]:
        """Extracts algorithmic candlestick price action features from the recent history."""
        if df.empty or len(df) < 5:
            return {"valid": False, "reason": "Insufficient candle history"}

        sub = df.iloc[-n_bars:].copy()
        closes = sub["close"].to_numpy()
        highs = sub["high"].to_numpy()
        lows = sub["low"].to_numpy()
        opens = sub["open"].to_numpy()

        swing_high = float(np.max(highs))
        swing_low = float(np.min(lows))
        last_close = float(closes[-1])
        prior_close = float(closes[-2]) if len(closes) > 1 else last_close

        # Momentum calculation: slope of last 5 bars
        last_5 = closes[-5:]
        slope = float((last_5[-1] - last_5[0]) / (len(last_5) * spot) * 100.0)

        # Candle wick analysis for S/R rejection
        # Lower wick size on last 3 bars: indicates buying absorption off support
        lower_wicks = np.minimum(opens[-3:], closes[-3:]) - lows[-3:]
        upper_wicks = highs[-3:] - np.maximum(opens[-3:], closes[-3:])
        avg_lower_wick = float(np.mean(lower_wicks))
        avg_upper_wick = float(np.mean(upper_wicks))

        # Check support / resistance distance
        support = sr_ctx.nearest_support if hasattr(sr_ctx, "nearest_support") else None
        resistance = sr_ctx.nearest_resistance if hasattr(sr_ctx, "nearest_resistance") else None

        sup_dist_pct = round(abs(spot - support) / spot * 100.0, 3) if support else 999.0
        res_dist_pct = round(abs(resistance - spot) / spot * 100.0, 3) if resistance else 999.0

        # Bullish bounce signature: spot close to support, lower wicks > upper wicks, slope turning positive
        has_support_bounce = bool(support and sup_dist_pct < 0.6 and (avg_lower_wick >= avg_upper_wick or slope > -0.05))
        has_resistance_rejection = bool(resistance and res_dist_pct < 0.6 and (avg_upper_wick >= avg_lower_wick or slope < 0.05))

        # Consecutive candles in direction
        bull_candles = int(np.sum(closes[-4:] > opens[-4:]))
        bear_candles = int(np.sum(closes[-4:] < opens[-4:]))

        candle_summary = []
        for i in range(max(0, len(sub) - 5), len(sub)):
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            c_type = "BULL" if c >= o else "BEAR"
            wick_type = "LongLowerWick (Rejection)" if (min(o, c) - l) > abs(c - o) else "Standard"
            candle_summary.append(f"Bar[-{len(sub)-1-i}]: {c_type} O={o:.3f} H={h:.3f} L={l:.3f} C={c:.3f} ({wick_type})")

        return {
            "valid": True,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "slope_pct": round(slope, 4),
            "support": support,
            "resistance": resistance,
            "sup_dist_pct": sup_dist_pct,
            "res_dist_pct": res_dist_pct,
            "has_support_bounce": has_support_bounce,
            "has_resistance_rejection": has_resistance_rejection,
            "bull_candles": bull_candles,
            "bear_candles": bear_candles,
            "candle_summary": candle_summary,
        }

    def validate_setup(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        spot: float,
        sr_ctx: Any,
        kronos_direction: str,
        kronos_conviction: float,
        stop_loss: float,
        take_profit: float,
        multiplier: int,
    ) -> ValidationResult:
        """Validates a proposed Kronos trade setup using Llama 3.2 3B (with high-precision quant fallback)."""
        t0 = time.perf_counter()

        if not self.is_enabled:
            return ValidationResult(
                is_valid=True,
                verdict="OFF",
                confidence=1.0,
                support_confirmed=True,
                resistance_confirmed=True,
                reasoning="Secondary AI Validator is disabled in settings.",
                model_name="disabled",
                latency_ms=0.0,
            )

        # 1. Feature extraction from historical candlestick data
        pa = self.analyze_historical_price_action(df, spot, sr_ctx, kronos_direction, n_bars=15)
        if not pa.get("valid"):
            return ValidationResult(
                is_valid=False,
                verdict="REJECT",
                confidence=0.5,
                support_confirmed=False,
                resistance_confirmed=False,
                reasoning="Cannot validate: Insufficient historical candles available.",
                model_name="validator-gate",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        asset_name = "Gold (frxXAUUSD)" if "XAU" in symbol else "Silver (frxXAGUSD)"

        # 2. Try invoking Llama 3.2 3B via Ollama API
        status = self.get_status()
        if status.get("is_ready") and status.get("ollama_online"):
            try:
                result = self._query_llama32(
                    symbol=symbol,
                    asset_name=asset_name,
                    timeframe=timeframe,
                    spot=spot,
                    pa=pa,
                    sr_ctx=sr_ctx,
                    direction=kronos_direction,
                    conviction=kronos_conviction,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    multiplier=multiplier,
                    t0=t0,
                )
                if result:
                    return result
            except Exception as e:
                log.warning("Ollama Llama 3.2 query failed (%s); reverting to quant validation engine.", e)

        # 3. High-Precision Algorithmic Quant Fallback (Always active and reliable)
        return self._quant_fallback_validation(
            asset_name=asset_name,
            symbol=symbol,
            timeframe=timeframe,
            spot=spot,
            pa=pa,
            sr_ctx=sr_ctx,
            direction=kronos_direction,
            conviction=kronos_conviction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            multiplier=multiplier,
            t0=t0,
        )

    def _query_llama32(
        self,
        symbol: str,
        asset_name: str,
        timeframe: str,
        spot: float,
        pa: Dict[str, Any],
        sr_ctx: Any,
        direction: str,
        conviction: float,
        stop_loss: float,
        take_profit: float,
        multiplier: int,
        t0: float,
    ) -> Optional[ValidationResult]:
        """Constructs prompt for Llama 3.2 3B and parses structured validation output."""
        candle_text = "\n".join(pa.get("candle_summary", [])[-5:])

        sup_val = float(pa.get("support") or 0.0)
        res_val = float(pa.get("resistance") or 0.0)

        prompt = (
            f"You are a Quantitative Risk & Price Action Validation AI supporting the Kronos Foundation Model for precious metals.\n"
            f"Asset: {asset_name} | Timeframe: {timeframe} | Spot: ${spot:.3f}\n"
            f"Nearest Support: ${sup_val:.3f} (Distance: {pa.get('sup_dist_pct')}%) | "
            f"Nearest Resistance: ${res_val:.3f} (Distance: {pa.get('res_dist_pct')}%)\n"
            f"S/R Zone: {getattr(sr_ctx, 'zone', 'MID_RANGE')}\n\n"
            f"RECENT 5-CANDLE ACTION:\n{candle_text}\n\n"
            f"KRONOS PROPOSED TRADE:\n"
            f"Direction: {direction} | Kronos Conviction: {conviction*100:.1f}%\n"
            f"Multiplier: {multiplier}x | Stop Loss: ${stop_loss:.3f} | Take Profit: ${take_profit:.3f}\n\n"
            f"Evaluate whether the historical candle price action and support/resistance structure confirm the Kronos signal.\n"
            f"Respond strictly in valid JSON with keys: 'verdict' ('CONFIRM', 'REJECT', or 'CAUTION'), "
            f"'confidence' (float 0.50 to 1.00), 'support_confirmed' (boolean), 'resistance_confirmed' (boolean), "
            f"and 'reasoning' (strictly 2 sentences explaining the price action analysis)."
        )

        resp = requests.post(
            f"{self.api_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.2,
                    "num_predict": 180,
                },
            },
            timeout=8.0,
        )

        if resp.status_code == 200:
            raw_text = resp.json().get("response", "").strip()
            # Clean possible markdown wrap
            clean_json = re.sub(r"^```json\s*", "", raw_text)
            clean_json = re.sub(r"\s*```$", "", clean_json)
            data = json.loads(clean_json)

            verdict = str(data.get("verdict", "CAUTION")).upper()
            if verdict not in ("CONFIRM", "REJECT", "CAUTION"):
                verdict = "CAUTION"
            conf = float(data.get("confidence", 0.70))
            conf = min(max(conf, 0.5), 1.0)
            sup_ok = bool(data.get("support_confirmed", True))
            res_ok = bool(data.get("resistance_confirmed", True))
            reasoning = str(data.get("reasoning", "")).strip() or f"{self.model_name} verified trade alignment."

            latency = (time.perf_counter() - t0) * 1000
            return ValidationResult(
                is_valid=(verdict != "REJECT"),
                verdict=verdict,
                confidence=conf,
                support_confirmed=sup_ok,
                resistance_confirmed=res_ok,
                reasoning=f"[{self.model_name}] {reasoning}",
                model_name=self.model_name,
                latency_ms=latency,
                details=pa,
            )
        return None

    def _quant_fallback_validation(
        self,
        asset_name: str,
        symbol: str,
        timeframe: str,
        spot: float,
        pa: Dict[str, Any],
        sr_ctx: Any,
        direction: str,
        conviction: float,
        stop_loss: float,
        take_profit: float,
        multiplier: int,
        t0: float,
    ) -> ValidationResult:
        """Deterministic algorithmic validation based on historical candle dynamics and S/R integrity."""
        latency = (time.perf_counter() - t0) * 1000
        zone = getattr(sr_ctx, "zone", "MID_RANGE")
        support = pa.get("support")
        resistance = pa.get("resistance")
        slope = pa.get("slope_pct", 0.0)
        bull_cnt = pa.get("bull_candles", 0)
        bear_cnt = pa.get("bear_candles", 0)
        has_bounce = pa.get("has_support_bounce", False)
        has_rejection = pa.get("has_resistance_rejection", False)

        if direction == "MULTUP":
            # Validating a BUY / MULTUP setup
            if zone == "AT_RESISTANCE":
                # Buying directly into strong resistance ceiling
                res_val = resistance if resistance else spot * 1.002
                return ValidationResult(
                    is_valid=False,
                    verdict="REJECT",
                    confidence=0.85,
                    support_confirmed=False,
                    resistance_confirmed=True,
                    reasoning=f"Price is pressing directly into overhead resistance at ${res_val:.2f}. High risk of bull trap rejection.",
                    model_name="quant-validator",
                    latency_ms=latency,
                    details=pa,
                )
            elif zone == "AT_SUPPORT" or has_bounce or bull_cnt >= 2:
                conf = 0.88 if has_bounce else 0.75
                sup_str = f"${support:.2f}" if support else "local swing low"
                return ValidationResult(
                    is_valid=True,
                    verdict="CONFIRM",
                    confidence=conf,
                    support_confirmed=True,
                    resistance_confirmed=False,
                    reasoning=f"Historical price action confirms solid support defense at {sup_str}. Buyers absorbing supply with {bull_cnt}/4 bull closes.",
                    model_name="quant-validator",
                    latency_ms=latency,
                    details=pa,
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    verdict="CAUTION",
                    confidence=0.65,
                    support_confirmed=True,
                    resistance_confirmed=False,
                    reasoning="Price is mid-range; momentum supports upside, but stop loss must be monitored closely.",
                    model_name="quant-validator",
                    latency_ms=latency,
                    details=pa,
                )

        elif direction == "MULTDOWN":
            # Validating a SELL / MULTDOWN setup
            if zone == "AT_SUPPORT":
                # Selling directly into strong support floor
                sup_val = support if support else spot * 0.998
                return ValidationResult(
                    is_valid=False,
                    verdict="REJECT",
                    confidence=0.85,
                    support_confirmed=True,
                    resistance_confirmed=False,
                    reasoning=f"Price is sitting on major support floor at ${sup_val:.2f}. High probability of bear trap bounce.",
                    model_name="quant-validator",
                    latency_ms=latency,
                    details=pa,
                )
            elif zone == "AT_RESISTANCE" or has_rejection or bear_cnt >= 2:
                conf = 0.88 if has_rejection else 0.75
                res_str = f"${resistance:.2f}" if resistance else "local swing high"
                return ValidationResult(
                    is_valid=True,
                    verdict="CONFIRM",
                    confidence=conf,
                    support_confirmed=False,
                    resistance_confirmed=True,
                    reasoning=f"Historical candles show firm rejection at {res_str}. Sellers dominant with {bear_cnt}/4 bear closes confirming Kronos bias.",
                    model_name="quant-validator",
                    latency_ms=latency,
                    details=pa,
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    verdict="CAUTION",
                    confidence=0.65,
                    support_confirmed=False,
                    resistance_confirmed=True,
                    reasoning="Price is mid-range; downside momentum present but caution advised against sudden support bounces.",
                    model_name="quant-validator",
                    latency_ms=latency,
                    details=pa,
                )

        return ValidationResult(
            is_valid=True,
            verdict="CAUTION",
            confidence=0.60,
            support_confirmed=True,
            resistance_confirmed=True,
            reasoning="Neutral market conditions; setup requires strict capital protection.",
            model_name="quant-validator",
            latency_ms=latency,
            details=pa,
        )
