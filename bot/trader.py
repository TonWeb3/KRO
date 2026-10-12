"""Main Multi-Asset Autonomous Trading Loop for Gold and Silver Multipliers.

Coordinates:
- Multi-Asset Candle Buffers across 5m, 15m, 30m, and 1h for Gold & Silver
- Balance-Adaptive Timeframe Selection (low balance focuses on 5m micro-stops)
- Support & Resistance Detection & Dynamic Multiplier Sizing
- High-Impact Macro Economic News Breakout Strategy
- Global Financial Session Clock & Overlap Liquidity
- Concurrent Execution & Real-Time Position Tracking for Both Metals
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .config import CFG, CONFIG_PATH, Config
from .deriv_client import DerivClient
from .engine import INTERVAL_MINUTES, KronosEngine
from .news_engine import NewsEngine, NewsState
from .session_clock import SessionInfo, get_session_info
from .signals import FLAT, MULTDOWN, MULTUP, MultiplierTradePlan, build_multiplier_plan
from .sr_levels import SRContext, compute_sr_context
from .storage import Store
from .validator_model import ValidatorManager
from .position_monitor import PositionMonitor

log = logging.getLogger(__name__)

NUM_COLS = ["open", "high", "low", "close", "volume", "amount"]


SEC_TO_INTERVAL = {300: "5m", 900: "15m", 1800: "30m", 3600: "1h", 14400: "4h"}


class MultiAssetCandleStore:
    """Stores rolling closed candles for multiple symbols and timeframes,
    updated in real-time via persistent Deriv WebSocket OHLC streaming.
    """

    def __init__(self, client: DerivClient, symbols: List[str], intervals: List[str], maxlen: int = 2000,
                 on_candle_closed: Optional[Callable[[str, str], None]] = None):
        self.client = client
        self.symbols = symbols
        self.intervals = intervals
        self.maxlen = maxlen
        self.on_candle_closed = on_candle_closed
        # _frames[sym][interval] -> pd.DataFrame of strictly CLOSED historical candles
        self._frames: Dict[str, Dict[str, pd.DataFrame]] = {s: {} for s in symbols}
        # _forming_bars[sym][interval] -> dict of active real-time candle
        self._forming_bars: Dict[str, Dict[str, dict]] = {s: {} for s in symbols}
        self._lock = asyncio.Lock()
        self._backfill_lock = asyncio.Lock()
        self.bar_closed: Dict[str, Dict[str, asyncio.Event]] = {
            s: {i: asyncio.Event() for i in intervals} for s in symbols
        }
        self._last_stream_ts: Dict[str, Dict[str, float]] = {
            s: {i: 0.0 for i in intervals} for s in symbols
        }
        # Wire live OHLC packet dispatcher from Deriv WebSocket
        self.client.on_ohlc = self.handle_ohlc_update

    def _to_df(self, raw_candles: List[dict]) -> pd.DataFrame:
        if not raw_candles:
            return pd.DataFrame(columns=["timestamps", "open_time", "close_time"] + NUM_COLS)
        df = pd.DataFrame(raw_candles)
        df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in NUM_COLS:
            df[c] = df[c].astype(float)
        now_ms = int(time.time() * 1000)
        if len(df) and df["close_time"].iloc[-1] >= now_ms:
            df = df.iloc[:-1]
        return df[["timestamps", "open_time", "close_time"] + NUM_COLS].drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    def handle_ohlc_update(self, ohlc: Dict[str, Any]) -> None:
        """Process real-time streaming OHLC message pushed directly by Deriv WebSocket."""
        sym = ohlc.get("symbol")
        if sym not in self.symbols:
            return
        g = int(ohlc.get("granularity") or 0)
        iv = SEC_TO_INTERVAL.get(g)
        if not iv or iv not in self.intervals:
            return

        open_epoch = int(ohlc.get("open_time") or 0)
        if open_epoch <= 0:
            return
        cur_open_ms = open_epoch * 1000
        cur_close_ms = cur_open_ms + (g * 1000) - 1

        open_p = float(ohlc["open"])
        high_p = float(ohlc["high"])
        low_p = float(ohlc["low"])
        close_p = float(ohlc["close"])

        # Detect candle close when open_time advances past previous forming bar
        fb = self._forming_bars.get(sym, {}).get(iv)
        if fb is not None and fb.get("open_time") is not None and fb["open_time"] < cur_open_ms:
            # *** THE PREVIOUS FORMING BAR HAS COMPLETED & CLOSED! ***
            closed_bar = {
                "timestamps": pd.to_datetime(fb["open_time"], unit="ms"),
                "open_time": int(fb["open_time"]),
                "close_time": int(fb["close_time"]),
                "open": float(fb["open"]),
                "high": float(fb["high"]),
                "low": float(fb["low"]),
                "close": float(fb["close"]),
                "volume": float(fb.get("volume", 0.0) or 0.0),
                "amount": float(fb.get("amount", 0.0) or 0.0),
            }
            fresh = self._append(sym, iv, closed_bar)
            if fresh:
                self.bar_closed[sym][iv].set()
                log.info("REAL-TIME STREAM BAR CLOSED: %s %s at spot %.3f (bar_open=%s) -> 2000-candle history updated",
                         sym, iv, closed_bar["close"], closed_bar["timestamps"])
                if self.on_candle_closed:
                    try:
                        self.on_candle_closed(sym, iv)
                    except Exception as e:
                        log.debug("on_candle_closed callback error: %s", e)

        # Update active forming bar in real time with exact live values
        self._forming_bars[sym][iv] = {
            "timestamps": pd.to_datetime(cur_open_ms, unit="ms"),
            "open_time": cur_open_ms,
            "close_time": cur_close_ms,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": float(ohlc.get("volume", 0.0) or 0.0),
            "amount": float(ohlc.get("amount", 0.0) or 0.0),
        }
        self._last_stream_ts[sym][iv] = time.time()

    async def backfill(self) -> None:
        """Fetches 2000 historical candles for each timeframe from the start
        and establishes persistent WebSocket OHLC streaming for real-time updates.
        """
        if self._backfill_lock.locked():
            return
        async with self._backfill_lock:
            for sym in self.symbols:
                for iv in self.intervals:
                    sec = INTERVAL_MINUTES.get(iv, 5) * 60
                    try:
                        # Request 2000 historical candles (with pagination) for complete S/R structural awareness
                        count_bars = min(2000, self.maxlen)
                        raw = await self.client.fetch_candles(sym, granularity=sec, count=count_bars)
                        now_ms = int(time.time() * 1000)
                        if raw and raw[-1].get("close_time", 0) >= now_ms:
                            self._forming_bars[sym][iv] = dict(raw[-1])
                        df = self._to_df(raw)
                        self._frames[sym][iv] = df
                        self._last_stream_ts[sym][iv] = time.time()
                        log.info("Deep History Loaded for %s %s: %d closed bars (up to 2000 candles for S/R)",
                                 sym, iv, len(df))

                        # Subscribe to continuous real-time WebSocket OHLC stream
                        await self.client.subscribe_candles_stream(sym, granularity=sec)
                    except Exception as e:
                        log.warning("Backfill/subscribe failed for %s %s: %s", sym, iv, e)
                    await asyncio.sleep(0.05)

    def get(self, symbol: str, interval: str, include_forming: bool = False) -> pd.DataFrame:
        df = self._frames.get(symbol, {}).get(interval, pd.DataFrame()).copy()
        if not include_forming or df.empty:
            return df

        fb = self._forming_bars.get(symbol, {}).get(interval)
        if fb and fb.get("open_time") is not None:
            spot = self.client.get_spot(symbol)
            forming_close = spot if spot is not None else fb["close"]
            forming_high = max(fb["high"], forming_close)
            forming_low = min(fb["low"], forming_close)
            forming_row = {
                "timestamps": fb.get("timestamps") if fb.get("timestamps") is not None else pd.to_datetime(fb["open_time"], unit="ms"),
                "open_time": fb["open_time"],
                "close_time": fb["close_time"],
                "open": fb["open"],
                "high": forming_high,
                "low": forming_low,
                "close": forming_close,
                "volume": fb.get("volume", 0.0),
                "amount": fb.get("amount", 0.0),
            }
            df = pd.concat([df, pd.DataFrame([forming_row])], ignore_index=True)
        return df

    def _append(self, symbol: str, interval: str, row: dict) -> bool:
        df = self._frames.get(symbol, {}).get(interval)
        if df is None:
            return False
        if len(df) and row["open_time"] <= int(df["open_time"].iloc[-1]):
            return False
        self._frames[symbol][interval] = pd.concat(
            [df, pd.DataFrame([row])], ignore_index=True
        ).iloc[-self.maxlen:].reset_index(drop=True)
        return True

    async def stream_watchdog(self) -> None:
        """Passive streaming health watchdog. Zero HTTP/WS polling requests sent.
        Periodically monitors streaming health and logs status if markets pause.
        """
        while True:
            try:
                await asyncio.sleep(60.0)
                now = time.time()
                for sym in self.symbols:
                    last_tick = self.client.last_ts.get(sym, 0.0)
                    if last_tick > 0 and (now - last_tick) > 300.0:
                        log.warning("Stream Watchdog: No live ticks for %s in %.0fs (market paused or disconnected)",
                                    sym, now - last_tick)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.debug("Stream watchdog notice: %s", e)


@dataclass
class ActiveMultiplierContract:
    contract_id: int
    decision_id: int
    symbol: str                 # 'frxXAUUSD' or 'frxXAGUSD'
    direction: str              # 'MULTUP' | 'MULTDOWN'
    timeframe: str
    entry_spot: float
    current_spot: float
    multiplier: int
    stake: float
    stop_loss: float
    take_profit: float
    profit: float = 0.0
    profit_percentage: float = 0.0
    status: str = "open"
    opened_ts: float = field(default_factory=time.time)
    is_sold: bool = False
    highest_profit_pct: float = 0.0
    monitor_verdict: str = "HOLD"
    monitor_reason: str = "Position initialized"


class Bot:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.symbols = cfg.symbols or ["frxXAUUSD", "frxXAGUSD"]
        self.intervals = ["5m", "15m", "30m", "1h", "4h"]
        self.store = Store(cfg.db_path)
        self.client = DerivClient(
            cfg.app_id, cfg.token, symbols=self.symbols, mode=cfg.mode,
            on_connect=self._on_client_connected
        )
        self.candles = MultiAssetCandleStore(
            self.client, self.symbols, self.intervals,
            maxlen=2000,
            on_candle_closed=self._on_candle_closed
        )
        self.engine: Optional[KronosEngine] = None
        self.news_engine = NewsEngine()
        self.validator = ValidatorManager(self.cfg)
        self.position_monitor = PositionMonitor(self.cfg)
        self.macro_4h_fc: Dict[str, Optional[Any]] = {s: None for s in self.symbols}
        self._macro_4h_cache: Dict[str, Dict[str, Any]] = {s: {"ts": 0.0, "fc": None} for s in self.symbols}
        self._macro_inter_cache: Dict[str, Dict[str, Any]] = {s: {"ts": 0.0, "fc": None, "tf": ""} for s in self.symbols}
        self._last_config_mtime: float = os.path.getmtime(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else 0.0

        # Multi-asset runtime state
        self.forecasts: Dict[str, Dict[str, Any]] = {s: {} for s in self.symbols}
        self.sr_contexts: Dict[str, Optional[SRContext]] = {s: None for s in self.symbols}
        self.active_contracts: Dict[str, Optional[ActiveMultiplierContract]] = {s: None for s in self.symbols}
        self.last_plans: Dict[str, Optional[MultiplierTradePlan]] = {s: None for s in self.symbols}
        self.last_entry_ts: Dict[str, float] = {s: 0.0 for s in self.symbols}

        self.session_info: SessionInfo = get_session_info()
        self.news_state: NewsState = self.news_engine.evaluate()

        self.day_key = time.strftime("%Y-%m-%d", time.gmtime())
        self.day_pnl = 0.0
        self.day_trades = 0
        self.day_start_balance: Optional[float] = None
        self.trading_enabled = False
        self.halted_reason = "stopped (click Start to trade)"
        self.last_error = ""
        self.started_at = time.time()
        self.status = "starting"
        self.busy = ""
        self.eval_trigger = asyncio.Event()
        self.running = True

    def get_spot(self, symbol: str) -> Optional[float]:
        p = self.client.get_spot(symbol)
        if p is not None and p > 0:
            return p
        for iv in ["5m", "15m", "30m", "1h", "4h"]:
            df = self.candles.get(symbol, iv)
            if df is not None and len(df) and "close" in df.columns:
                return float(df["close"].iloc[-1])
        return None

    def _on_candle_closed(self, symbol: str, interval: str) -> None:
        """Invoked the exact millisecond a live candle completes and officially closes via WebSocket.
        The completed bar is added to the rolling 2,000-candle history (oldest bar discarded if >2000),
        and fresh model analysis is triggered immediately.
        """
        bal = self.client.balance or 1000.0
        act_tf = self.cfg.get_active_timeframe(bal)

        # Invalidate cached macro forecasts so closed higher timeframe bars immediately trigger fresh analysis
        if interval == "4h":
            self._macro_4h_cache.pop(symbol, None)
        elif interval in ("30m", "1h"):
            self._macro_inter_cache.pop(symbol, None)

        log.info("WebSocket candle closed for %s [%s] -> Rolling 2000-candle store updated & triggering immediate fresh model analysis",
                 symbol, interval)
        self.eval_trigger.set()

    async def _on_client_connected(self) -> None:
        """Invoked automatically whenever Deriv WS connects or reconnects."""
        log.info("Deriv WS online/reconnected -> backfilling candle store and refreshing evaluations")
        try:
            await self.candles.backfill()
            self.eval_trigger.set()
        except Exception as e:
            log.warning("Post-connect backfill error: %s", e)

    async def start(self) -> None:
        self.status = "connecting deriv"
        asyncio.create_task(self.client.start())

        while not self.client.connected and self.running:
            await asyncio.sleep(0.2)

        self.status = "backfilling candles"
        await self.candles.backfill()

        # Compute immediate initial S/R contexts across all timeframes from backfilled candles
        try:
            bal = self.client.balance or 1000.0
            act_tf = self.cfg.get_active_timeframe(bal)
            for sym in self.symbols:
                spot = self.client.get_spot(sym) or 0.0
                df_5m = self.candles.get(sym, "5m", include_forming=True)
                df_15m = self.candles.get(sym, "15m", include_forming=True)
                df_30m = self.candles.get(sym, "30m", include_forming=True)
                df_1h = self.candles.get(sym, "1h", include_forming=True)
                df_4h = self.candles.get(sym, "4h", include_forming=True)
                if spot == 0.0 and len(df_5m):
                    spot = float(df_5m["close"].iloc[-1])
                self.sr_contexts[sym] = compute_sr_context(
                    sym, spot, df_5m, df_15m, df_30m, df_1h, df_4h, None, None,
                    entry_fc=None, balance=bal, entry_timeframe=act_tf
                )
        except Exception as e:
            log.warning("Initial S/R context computation error: %s", e)

        self.status = "loading model"
        log.info("Loading Kronos %s ...", self.cfg.model_id)
        self.engine = await asyncio.to_thread(
            KronosEngine, self.cfg.model_id, self.cfg.tokenizer_id,
            self.cfg.device, self.cfg.max_context
        )

        self.status = "ready"
        log.info("Kronos Multiplier Bot ready for Gold & Silver. Mode=%s Risk=%s %s (%s)",
                 self.cfg.mode, self.cfg.risk_value,
                 "%" if self.cfg.risk_type == "percent" else self.cfg.currency, self.cfg.risk_type)

    async def stop(self) -> None:
        self.running = False
        self.client.close()

    async def run(self) -> None:
        await self.start()
        tasks = [
            asyncio.create_task(self.candles.stream_watchdog(), name="stream_watchdog"),
            asyncio.create_task(self._entry_loop(), name="entry"),
            asyncio.create_task(self._manage_loop(), name="manage"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await self.stop()

    async def _forecast(self, symbol: str, interval: str, horizon: int, n_paths: int, include_forming: bool = False):
        df = self.candles.get(symbol, interval, include_forming=include_forming)
        if len(df) < 50:
            return None
        # Decouple S/R history from model input: S/R engine uses all 2000 candles for deep structural memory,
        # but Kronos is fed strictly its optimal lookback (e.g. 200-288 bars) to prevent transformer sequence overload
        model_lookback = min(self.cfg.lookback, 288, len(df))
        return await asyncio.to_thread(
            self.engine.forecast, df, interval, horizon, n_paths,
            model_lookback, symbol, self.cfg.temperature,
            self.cfg.top_k, self.cfg.top_p, self.cfg.max_batch
        )

    async def _entry_loop(self) -> None:
        # Initial preview forecast for both metals
        await asyncio.sleep(2)
        await self._evaluate_all(preview=True)

        while self.running:
            # Re-evaluate every 30 seconds or immediately when triggered (e.g. contract settled)
            try:
                await asyncio.wait_for(self.eval_trigger.wait(), timeout=30.0)
                self.eval_trigger.clear()
            except asyncio.TimeoutError:
                pass

            if not self.running:
                break

            try:
                await self._evaluate_all()
            except Exception as e:
                self.last_error = f"eval: {e}"
                log.exception("Multiplier evaluation failed")

    async def _evaluate_all(self, preview: bool = False) -> None:
        # Detect if config.json was modified on disk (via settings UI or external edit) and live-reload
        try:
            mtime = os.path.getmtime(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else 0.0
            if mtime > self._last_config_mtime and self._last_config_mtime > 0:
                log.info("Detected change in %s -> Live reloading config parameters into running engine...", CONFIG_PATH)
                fresh = Config.load(CONFIG_PATH)
                from dataclasses import fields as dc_fields
                for f in dc_fields(fresh):
                    setattr(self.cfg, f.name, getattr(fresh, f.name))
                self._last_config_mtime = mtime
            elif self._last_config_mtime == 0.0:
                self._last_config_mtime = mtime
        except Exception as e:
            log.debug("Config live reload check error: %s", e)

        self.session_info = get_session_info()
        self.news_state = self.news_engine.evaluate()
        balance = self.client.balance

        # Full adaptive timeframe hierarchy based on current account balance
        active_tf, inter_tf, active_macro_tf = self.cfg.get_timeframe_hierarchy(balance)
        self.busy = f"evaluating Gold & Silver on {active_tf} (inter {inter_tf}, macro {active_macro_tf})"

        # Compute multi-timeframe S/R levels immediately before slower forecast models run
        for sym in self.symbols:
            try:
                spot = self.client.get_spot(sym) or 0.0
                df_5m = self.candles.get(sym, "5m", include_forming=True)
                df_15m = self.candles.get(sym, "15m", include_forming=True)
                df_30m = self.candles.get(sym, "30m", include_forming=True)
                df_1h = self.candles.get(sym, "1h", include_forming=True)
                df_4h = self.candles.get(sym, "4h", include_forming=True)
                if spot == 0.0 and len(df_5m):
                    spot = float(df_5m["close"].iloc[-1])
                self.sr_contexts[sym] = compute_sr_context(
                    sym, spot, df_5m, df_15m, df_30m, df_1h, df_4h, None, None,
                    entry_fc=None, balance=balance, entry_timeframe=active_tf
                )
            except Exception as e:
                log.warning("Fast S/R refresh error for %s: %s", sym, e)

        # 1. Obtain 4h and adaptive macro forecasts with 15-minute caching to eliminate CPU overload
        macro_4h_fc: Dict[str, Optional[Any]] = {}
        macro_inter_fc: Dict[str, Optional[Any]] = {}
        now_ts = time.time()
        macro_ttl = 900.0  # 15 minutes

        for sym in self.symbols:
            c4 = self._macro_4h_cache.get(sym, {})
            if c4.get("fc") is not None and (now_ts - c4.get("ts", 0.0)) < macro_ttl:
                macro_4h_fc[sym] = c4["fc"]
            else:
                try:
                    m4 = await self._forecast(sym, "4h", 12, self.cfg.n_paths_regime)
                    self._macro_4h_cache[sym] = {"ts": now_ts, "fc": m4}
                    macro_4h_fc[sym] = m4
                except Exception as e:
                    log.warning("4h macro forecast failed for %s: %s", sym, e)
                    macro_4h_fc[sym] = c4.get("fc")

            ci = self._macro_inter_cache.get(sym, {})
            if ci.get("fc") is not None and ci.get("tf") == active_macro_tf and (now_ts - ci.get("ts", 0.0)) < macro_ttl:
                macro_inter_fc[sym] = ci["fc"]
            else:
                try:
                    mi = await self._forecast(sym, active_macro_tf, 12, self.cfg.n_paths_regime)
                    self._macro_inter_cache[sym] = {"ts": now_ts, "fc": mi, "tf": active_macro_tf}
                    macro_inter_fc[sym] = mi
                except Exception as e:
                    log.warning("%s macro forecast failed for %s: %s", active_macro_tf, sym, e)
                    macro_inter_fc[sym] = ci.get("fc")

        self.macro_4h_fc = macro_4h_fc

        # 2. Generate entry forecasts and S/R context for both symbols
        entry_fc: Dict[str, Optional[Any]] = {}
        plans: Dict[str, MultiplierTradePlan] = {}

        for sym in self.symbols:
            try:
                efc = await self._forecast(sym, active_tf, self.cfg.entry_pred_len, self.cfg.n_paths, include_forming=True)
                entry_fc[sym] = efc
                if efc:
                    self.forecasts[sym][active_tf] = efc
                    self.store.log_forecast(efc)

                # Support & Resistance detection across all deep multi-timeframe candles
                df_5m = self.candles.get(sym, "5m", include_forming=True)
                df_15m = self.candles.get(sym, "15m", include_forming=True)
                df_30m = self.candles.get(sym, "30m", include_forming=True)
                df_1h = self.candles.get(sym, "1h", include_forming=True)
                df_4h = self.candles.get(sym, "4h", include_forming=True)
                lo_b, hi_b = efc.band(10, 90) if efc else (None, None)
                fc_lo = float(lo_b[0]) if lo_b is not None and len(lo_b) else None
                fc_hi = float(hi_b[0]) if hi_b is not None and len(hi_b) else None

                spot = self.client.get_spot(sym) or (efc.last_close if efc else 0.0)
                active_tf = self.cfg.get_active_timeframe(balance)
                sr_ctx = compute_sr_context(
                    sym, spot, df_5m, df_15m, df_30m, df_1h, df_4h, fc_lo, fc_hi,
                    entry_fc=efc, balance=balance, entry_timeframe=active_tf
                )
                self.sr_contexts[sym] = sr_ctx

            except Exception as e:
                log.warning("Forecast / SR calculation failed for %s: %s", sym, e)
                entry_fc[sym] = None

        # Check cross-asset directional agreement between Gold and Silver
        gold_up = entry_fc.get("frxXAUUSD").upside_prob if entry_fc.get("frxXAUUSD") else 0.5
        silver_up = entry_fc.get("frxXAGUSD").upside_prob if entry_fc.get("frxXAGUSD") else 0.5
        cross_agree = (gold_up > 0.5 and silver_up > 0.5) or (gold_up < 0.5 and silver_up < 0.5)

        # 3. Build trade plans and execute qualified setups
        for sym in self.symbols:
            ef = entry_fc.get(sym)
            if not ef:
                continue

            spot = self.client.get_spot(sym) or ef.last_close

            # Secondary AI Validator review (RayCodes Llama 3.2 3B Support Model)
            v_res = None
            if self.cfg.enable_validator_model and self.sr_contexts.get(sym):
                tf = active_tf
                hist_df = self.candles.get(sym, tf, include_forming=True).tail(40)
                k_bias = MULTUP if ef.upside_prob > 0.5 else MULTDOWN
                sr_c = self.sr_contexts[sym]
                tentative_sl = sr_c.nearest_support * 0.9995 if k_bias == MULTUP and sr_c.nearest_support else spot * 0.995
                tentative_tp = sr_c.nearest_resistance * 1.0005 if k_bias == MULTUP and sr_c.nearest_resistance else spot * 1.005
                mult, _, _ = self.cfg.calculate_multiplier_sizing(balance, spot, tentative_sl)
                v_res = await asyncio.to_thread(
                    self.validator.validate_setup,
                    symbol=sym,
                    timeframe=tf,
                    df=hist_df,
                    spot=spot,
                    sr_ctx=sr_c,
                    kronos_direction=k_bias,
                    kronos_conviction=max(ef.upside_prob, 1.0 - ef.upside_prob),
                    stop_loss=tentative_sl,
                    take_profit=tentative_tp,
                    multiplier=mult,
                )

            candles_tf = self.candles.get(sym, active_tf, include_forming=True)
            plan = build_multiplier_plan(
                self.cfg, sym, ef, macro_inter_fc.get(sym), self.sr_contexts[sym],
                self.session_info, self.news_state, spot, balance=balance,
                cross_confirm=cross_agree, validator_res=v_res,
                macro_4h_fc=macro_4h_fc.get(sym),
                candles_df=candles_tf
            )
            self.last_plans[sym] = plan

            cur_contract = self.active_contracts.get(sym)
            is_flip = False

            if cur_contract is not None:
                # Active position currently open on this asset
                if plan.tradeable and plan.direction in (MULTUP, MULTDOWN) and plan.direction != cur_contract.direction:
                    is_flip = True
                    log.warning("DIRECTION REVERSAL DETECTED on %s: Active %s -> Qualified %s (%s). Executing direction flip.",
                                sym, cur_contract.direction, plan.direction, plan.quality_grade)
                elif cur_contract.direction == plan.direction:
                    # Same direction: Still says Buy while in Buy (or Sell while in Sell) -> LEAVE IT TO RUN!
                    log.info("%s active %s position intact | Analysis supports continuation -> LEAVING WINNER TO RUN",
                             sym, cur_contract.direction)
                    continue
                else:
                    # Plan is FLAT: No opposite trigger -> LEAVE IT TO RUN!
                    log.info("%s active %s position intact | Neutral holding -> LEAVING TO RUN",
                             sym, cur_contract.direction)
                    continue

            block = "preview" if preview else self._risk_block(sym, is_flip=is_flip)
            if block and plan.direction != FLAT:
                plan.direction = FLAT
                plan.reason = f"blocked: {block}"

            taken = plan.tradeable and not preview
            bar_tag = ef.last_ts.isoformat()
            decision_id = self.store.log_decision(plan, bar_tag, taken)

            log.info("%s Multiplier | %s %s | Grade %s | Spot %.3f SL %.3f (RRR %.1f) | Mult %dx Stake $%.2f | %s",
                     sym, plan.timeframe, plan.direction, plan.quality_grade,
                     plan.spot, plan.stop_loss, plan.rrr, plan.multiplier, plan.stake, plan.reason)

            if taken:
                if is_flip and cur_contract is not None:
                    # Close the active opposite contract immediately
                    cid = cur_contract.contract_id
                    log.warning("CLOSING ACTIVE %s #%s to immediately reverse into %s", cur_contract.direction, cid, plan.direction)
                    sell_res = await self.client.sell(cid)
                    if sell_res.get("ok"):
                        exit_spot = cur_contract.current_spot or spot
                        profit = cur_contract.profit
                        status = "won" if profit > 0 else "lost"
                        self.store.close_trade(cid, exit_spot, profit, status)
                        self.day_pnl += profit
                        await self.client.fetch_balance()
                        self.active_contracts[sym] = None
                        log.info("Closed %s #%s successfully (P&L $%.2f). Opening reverse %s contract now...",
                                 cur_contract.direction, cid, profit, plan.direction)

                await self._execute_multiplier(plan, decision_id)

        self.busy = ""

    def _roll_day(self) -> None:
        k = time.strftime("%Y-%m-%d", time.gmtime())
        bal = self.client.balance if (self.client.balance is not None and self.client.balance > 0) else None
        if self.day_start_balance is None and bal is not None:
            self.day_start_balance = bal

        if k != self.day_key:
            self.day_key = k
            self.day_pnl = 0.0
            self.day_trades = 0
            self.day_start_balance = bal or self.day_start_balance or 100.0
            if self.halted_reason.startswith("daily"):
                self.halted_reason = ""

    def _risk_block(self, symbol: str, is_flip: bool = False) -> str:
        self._roll_day()
        if not self.trading_enabled:
            return "bot stopped (click Start)"
        if self.halted_reason:
            return self.halted_reason
        if not is_flip and self.active_contracts.get(symbol) is not None:
            return f"{symbol} multiplier already open"

        active_count = sum(1 for s, c in self.active_contracts.items() if c is not None and (not is_flip or s != symbol))
        if active_count >= self.cfg.max_concurrent:
            return "max concurrent trades reached"

        start_bal = self.day_start_balance or self.client.balance or 100.0
        max_loss_usd = (self.cfg.max_daily_loss_pct / 100.0) * start_bal
        if self.day_pnl <= -abs(max_loss_usd):
            self.halted_reason = f"daily loss limit ({self.cfg.max_daily_loss_pct}% / -${abs(max_loss_usd):.2f})"
            return self.halted_reason
        if self.day_trades >= self.cfg.max_trades_per_day:
            return "daily trade cap"

        if not is_flip:
            sec_since = time.time() - self.last_entry_ts.get(symbol, 0.0)
            cb = max(0, getattr(self.cfg, "cooldown_bars", 1))
            cooldown_sec = INTERVAL_MINUTES.get(self.cfg.get_active_timeframe(self.client.balance), 5) * 60 * cb
            if cooldown_sec > 0 and sec_since < cooldown_sec:
                return "cooldown"

        if self.client.balance is not None and self.client.balance <= 0:
            return "insufficient balance"
        return ""

    async def _execute_multiplier(self, plan: MultiplierTradePlan, decision_id: int) -> None:
        if not self.client.authorized:
            log.warning("Cannot execute multiplier: Deriv client not authorized")
            return

        # 1. Price Multiplier proposal with attached Stop Loss ($) and Take Profit ($)
        prop = await self.client.get_multiplier_proposal(
            symbol=plan.symbol,
            contract_type=plan.direction,
            amount=plan.stake,
            multiplier=plan.multiplier,
            stop_loss_usd=plan.stop_loss_usd,
            take_profit_usd=plan.take_profit_usd,
            cancellation="15m" if self.cfg.use_deal_cancellation else None
        )
        if not prop.get("ok"):
            log.warning("Multiplier proposal rejected (%s): %s", plan.symbol, prop.get("error"))
            return

        proposal_id = prop["id"]
        ask_price = prop["ask_price"]

        # 2. Buy contract
        buy_res = await self.client.buy(proposal_id, ask_price)
        if not buy_res.get("ok"):
            log.warning("Multiplier buy failed (%s): %s", plan.symbol, buy_res.get("error"))
            return

        contract_id = int(buy_res["contract_id"])
        spot_exec = prop.get("spot") or plan.spot
        self.store.open_trade(decision_id, contract_id, plan, ask_price, self.cfg.mode, entry_spot=spot_exec)

        self.active_contracts[plan.symbol] = ActiveMultiplierContract(
            contract_id=contract_id,
            decision_id=decision_id,
            symbol=plan.symbol,
            direction=plan.direction,
            timeframe=plan.timeframe,
            entry_spot=spot_exec,
            current_spot=spot_exec,
            multiplier=plan.multiplier,
            stake=ask_price,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
            opened_ts=time.time()
        )
        self.last_entry_ts[plan.symbol] = time.time()
        self.day_trades += 1

        log.info("OPENED MULTIPLIER %s %s %dx | Contract #%s | Stake $%.2f | SL Spot %.3f (Max Risk $%.2f) | TP Spot %.3f",
                 plan.symbol, plan.direction, plan.multiplier, contract_id,
                 ask_price, plan.stop_loss, plan.stop_loss_usd, plan.take_profit)

    async def _manage_loop(self) -> None:
        """Monitors open multiplier contracts in real-time until closed."""
        while self.running:
            await asyncio.sleep(1.0)
            for sym, contract in list(self.active_contracts.items()):
                if contract is None:
                    continue
                try:
                    cid = contract.contract_id
                    st = await self.client.contract_status(cid)
                    if not st.get("ok"):
                        continue

                    contract.current_spot = st.get("current_spot") or contract.current_spot
                    if st.get("entry_spot"):
                        contract.entry_spot = st["entry_spot"]
                    contract.profit = st.get("profit") or 0.0
                    contract.profit_percentage = st.get("profit_percentage") or 0.0
                    contract.status = st.get("status") or "open"

                    # Check if Deriv closed the contract (e.g. SL or TP hit)
                    if st.get("is_sold"):
                        exit_spot = st.get("current_spot") or contract.current_spot or 0.0
                        profit = st.get("profit") or 0.0
                        status = "won" if profit > 0 else "lost"

                        self.store.close_trade(cid, exit_spot, profit, status)
                        self.day_pnl += profit
                        log.info("SETTLED MULTIPLIER #%s (%s): %s | Profit: $%.2f | Balance: %s",
                                 cid, sym, status.upper(), profit, self.client.balance)

                        await self.client.fetch_balance()
                        self.active_contracts[sym] = None
                        self.last_entry_ts[sym] = time.time()  # Activate cooldown to prevent whipsaw revenge entry
                        self.eval_trigger.set()
                        continue

                    # Dynamic Position Monitor (autonomous HOLD vs CLOSE evaluation)
                    if getattr(self.cfg, "enable_position_monitor", True):
                        latest_fc = self.forecasts.get(sym, {}).get(contract.timeframe)
                        sr_c = self.sr_contexts.get(sym)
                        candles_df = self.candles.get(sym, contract.timeframe, include_forming=True)
                        macro_fc = self.macro_4h_fc.get(sym) or self._macro_inter_cache.get(sym, {}).get("fc")
                        verdict, reason = self.position_monitor.evaluate_position(
                            contract=contract,
                            current_spot=contract.current_spot,
                            latest_forecast=latest_fc,
                            sr_ctx=sr_c,
                            news=self.news_state,
                            candles_df=candles_df,
                            macro_fc=macro_fc,
                        )
                        contract.monitor_verdict = verdict
                        contract.monitor_reason = reason

                        if verdict == "CLOSE":
                            log.warning("POSITION MONITOR CLOSING [%s #%s]: %s", sym, cid, reason)
                            sell_res = await self.client.sell(cid)
                            if sell_res.get("ok"):
                                log.info("Position Monitor sold contract #%s (%s) successfully: %s", cid, sym, reason)
                                exit_spot = contract.current_spot
                                profit = contract.profit
                                status = "won" if profit > 0 else "lost"
                                self.store.close_trade(cid, exit_spot, profit, status)
                                self.day_pnl += profit
                                await self.client.fetch_balance()
                                self.active_contracts[sym] = None
                                self.last_entry_ts[sym] = time.time()  # Activate cooldown to prevent whipsaw revenge entry
                                self.eval_trigger.set()
                                continue

                except Exception as e:
                    self.last_error = f"manage ({sym}): {e}"
                    log.warning("Multiplier tracking error for %s: %s", sym, e)

    async def manual_close(self, symbol: str) -> Dict[str, Any]:
        """Manually closes an active multiplier position."""
        act = self.active_contracts.get(symbol)
        if not act:
            return {"ok": False, "error": f"No active contract for {symbol}"}
        res = await self.client.sell(act.contract_id)
        if res.get("ok"):
            exit_spot = self.client.get_spot(symbol) or act.current_spot
            sold_for = res.get("sold_for")
            profit = float(sold_for - act.stake) if sold_for is not None else act.profit
            status = "won" if profit > 0 else "lost"
            self.store.close_trade(act.contract_id, exit_spot, profit, status)
            self.day_pnl += profit
            self.active_contracts[symbol] = None
            self.last_entry_ts[symbol] = time.time()  # Activate cooldown
            await self.client.fetch_balance()
            self.eval_trigger.set()
            log.info("MANUALLY CLOSED MULTIPLIER #%s (%s): %s | Profit: $%.2f | Balance: %s",
                     act.contract_id, symbol, status.upper(), profit, self.client.balance)
        return res

    def state(self) -> dict:
        balance = self.client.balance
        active_tf = self.cfg.get_active_timeframe(balance)
        active_macro_tf = self.cfg.get_active_macro_timeframe(balance)

        # Serialized active contracts
        act_dict = {}
        for sym, c in self.active_contracts.items():
            if c:
                act_dict[sym] = {
                    "contract_id": c.contract_id,
                    "symbol": c.symbol,
                    "direction": c.direction,
                    "timeframe": c.timeframe,
                    "multiplier": c.multiplier,
                    "entry_spot": round(c.entry_spot or 0.0, 3),
                    "current_spot": round(c.current_spot or 0.0, 3),
                    "stake": round(c.stake or 0.0, 2),
                    "stop_loss": round(c.stop_loss or 0.0, 3),
                    "take_profit": round(c.take_profit or 0.0, 3),
                    "profit": round(c.profit or 0.0, 2),
                    "profit_percentage": round(c.profit_percentage or 0.0, 2),
                    "highest_profit_pct": round(float(getattr(c, "highest_profit_pct", 0.0) or 0.0), 2),
                    "monitor_verdict": getattr(c, "monitor_verdict", "HOLD"),
                    "monitor_reason": getattr(c, "monitor_reason", "Trade thesis intact"),
                    "status": c.status,
                }
            else:
                act_dict[sym] = None

        # S/R Contexts
        sr_dict = {sym: (self.sr_contexts[sym].to_dict() if self.sr_contexts.get(sym) else None) for sym in self.symbols}
        plans_dict = {sym: (self.last_plans[sym].to_dict() if self.last_plans.get(sym) else None) for sym in self.symbols}

        return {
            "status": self.status,
            "busy": self.busy,
            "symbols": self.symbols,
            "active_timeframe": active_tf,
            "timeframe_mode": self.cfg.timeframe_mode,
            "macro_tf": active_macro_tf,
            "mode": self.cfg.mode,
            "connected": self.client.connected,
            "authorized": self.client.authorized,
            "account_id": self.client.account_id or "—",
            "balance": round(balance, 2) if balance is not None else None,
            "currency": self.client.currency,
            "spots": {s: self.get_spot(s) for s in self.symbols},
            "day_pnl": round(self.day_pnl, 2),
            "day_trades": self.day_trades,
            "day_start_balance": round(self.day_start_balance, 2) if self.day_start_balance is not None else (round(balance, 2) if balance is not None else None),
            "max_daily_loss_pct": self.cfg.max_daily_loss_pct,
            "halted": self.halted_reason,
            "trading_enabled": self.trading_enabled and not bool(self.halted_reason),
            "last_error": self.last_error,
            "active_contracts": act_dict,
            "sr_contexts": sr_dict,
            "trade_plans": plans_dict,
            "session_clock": self.session_info.to_dict(),
            "macro_news": self.news_state.to_dict(),
            "validator": self.validator.get_status(),
            "risk_type": self.cfg.risk_type,
            "risk_value": self.cfg.risk_value,
            "macro_4h": {
                s: {
                    "bias": ("MULTUP" if self.macro_4h_fc[s].upside_prob > 0.52 else "MULTDOWN" if self.macro_4h_fc[s].upside_prob < 0.48 else "FLAT"),
                    "conviction": round(max(self.macro_4h_fc[s].upside_prob, 1.0 - self.macro_4h_fc[s].upside_prob), 3)
                } if self.macro_4h_fc.get(s) else None
                for s in self.symbols
            },
            "position_monitor": {
                "enabled": getattr(self.cfg, "enable_position_monitor", True),
                "close_on_reversal": getattr(self.cfg, "close_on_reversal", True),
                "reversal_prob_threshold": getattr(self.cfg, "reversal_prob_threshold", 0.65),
                "take_profit_early_pct": getattr(self.cfg, "take_profit_early_pct", 40.0),
                "protect_peak_profit": getattr(self.cfg, "protect_peak_profit", True),
            },
        }
