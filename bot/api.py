"""FastAPI backend serving Gold & Silver Multiplier Trading Bot and Dashboard."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
from fastapi import Body, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .config import CFG, coerce, settings_schema
from .trader import Bot

log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "templates")

app = FastAPI(title="Kronos Deriv Multipliers Bot")
BOT: Bot | None = None
BOT_TASK: asyncio.Task | None = None
BOOT_CFG: dict = {}
_CHART_CACHE: Dict[Tuple[str, str], Tuple[float, dict]] = {}


def _snapshot() -> dict:
    from .config import NEEDS_RESTART
    return {k: getattr(CFG, k) for k in NEEDS_RESTART}


def _restart_pending() -> list:
    return sorted(k for k, v in BOOT_CFG.items() if getattr(CFG, k) != v)


@app.on_event("startup")
async def _startup() -> None:
    global BOT, BOT_TASK, BOOT_CFG
    BOT = Bot()
    BOOT_CFG = _snapshot()
    BOT_TASK = asyncio.create_task(BOT.run(), name="bot")


@app.on_event("shutdown")
async def _shutdown() -> None:
    if BOT_TASK:
        BOT_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await BOT_TASK


def clean_for_json(obj):
    """Recursively converts NumPy and custom types to native Python JSON-serializable primitives."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [clean_for_json(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    return obj


@app.get("/api/state")
async def api_state():
    if BOT is None:
        return JSONResponse({"status": "starting"})
    try:
        raw_state = BOT.state()
        return clean_for_json(raw_state)
    except Exception as e:
        log.error("api_state serialization error: %s", e)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


def _build_chart_dict(symbol: str, interval: str, history: int = 250) -> Optional[dict]:
    """Extracts in-memory candles, forecast envelope, and S/R levels with 0ms overhead."""
    if BOT is None:
        return None

    now = time.time()
    # High-speed in-memory cache check (1.0s TTL)
    cached = _CHART_CACHE.get((symbol, interval))
    if cached is not None:
        c_ts, c_dict = cached
        if (now - c_ts) < 1.0:
            spot = BOT.client.get_spot(symbol)
            if spot:
                c_dict["spot"] = spot
            return c_dict

    try:
        # Retrieve already backfilled candle DataFrame from memory without blocking WebSocket calls
        df = BOT.candles.get(symbol, interval, include_forming=True)
        if df is not None and len(df):
            df = df.iloc[-history:]
        else:
            df = None
    except Exception as e:
        log.debug("Candle get exception for %s %s: %s", symbol, interval, e)
        df = None

    if df is None or not len(df):
        # Fall back to raw frames directly if get() encountered an issue
        try:
            raw_frame = getattr(BOT.candles, "_frames", {}).get(symbol, {}).get(interval)
            if raw_frame is not None and len(raw_frame):
                df = raw_frame.iloc[-history:]
        except Exception:
            df = None

    if df is None or not len(df):
        return None

    fc = BOT.forecasts.get(symbol, {}).get(interval)
    sr_ctx = BOT.sr_contexts.get(symbol)
    spot = BOT.client.get_spot(symbol) or float(df["close"].iloc[-1])

    srl_dict = sr_ctx.to_dict() if sr_ctx else {}
    if len(df) >= 35:
        try:
            from .sr_levels import compute_lux_pivots_and_breaks
            _, _, osc_arr, tf_breaks, r_zone, s_zone, _ = compute_lux_pivots_and_breaks(
                df, left_bars=15, right_bars=15, volume_thresh=20.0, timeframe=interval
            )
            if r_zone:
                srl_dict["active_resistance_zone"] = r_zone.to_dict()
                srl_dict["active_resistance"] = r_zone.high
            if s_zone:
                srl_dict["active_support_zone"] = s_zone.to_dict()
                srl_dict["active_support"] = s_zone.low
            if tf_breaks:
                srl_dict["break_events"] = [b.to_dict() for b in tf_breaks[-20:]]
            if len(osc_arr):
                srl_dict["volume_osc"] = round(float(osc_arr[-1]), 1)
        except Exception as e:
            log.debug("Timeframe SR calculation skipped for %s %s: %s", symbol, interval, e)

    out = {
        "symbol": symbol,
        "interval": interval,
        "forecast_pending": fc is None,
        "spot": spot,
        "hist_ts": [t.isoformat() for t in df["timestamps"]],
        "hist_close": [round(float(x), 3) for x in df["close"]],
        "hist_ohlcv": [
            [round(float(r.open), 3), round(float(r.high), 3),
             round(float(r.low), 3), round(float(r.close), 3),
             round(float(r.volume), 2)]
            for r in df.itertuples()
        ],
        "sr_levels": srl_dict if srl_dict else None,
    }

    if fc is not None:
        lo, hi = fc.band(10, 90)
        lo2, hi2 = fc.band(0, 100)
        out.update({
            "fc_ts": [t.isoformat() for t in fc.y_timestamps],
            "fc_mean": [round(float(x), 3) for x in fc.mean_path],
            "fc_lo": [round(float(x), 3) for x in lo],
            "fc_hi": [round(float(x), 3) for x in hi],
            "fc_min": [round(float(x), 3) for x in lo2],
            "fc_max": [round(float(x), 3) for x in hi2],
            "summary": fc.summary(),
            "fc_ohlc": [
                [round(float(v), 3) for v in row]
                for row in np.median(fc.paths[:, :, :4], axis=0)
            ],
        })

    cleaned_out = clean_for_json(out)
    _CHART_CACHE[(symbol, interval)] = (now, cleaned_out)
    return cleaned_out


@app.get("/api/chart")
async def api_chart(symbol: str = Query("frxXAUUSD"), interval: str = Query("5m"), history: int = Query(250)):
    """Candlestick history + Kronos forecast envelope + S/R levels for requested symbol and timeframe."""
    if BOT is None:
        return JSONResponse({"status": "starting"})

    chart_data = _build_chart_dict(symbol, interval, history=history)
    if chart_data is not None:
        return chart_data

    return JSONResponse({"status": "no data", "forecast_pending": True})


@app.get("/api/charts/all")
async def api_charts_all(history: int = Query(250)):
    """Bulk candlestick history + Kronos forecasts + S/R levels for all metals and timeframes in a single fast response."""
    if BOT is None:
        return JSONResponse({"status": "starting"})

    symbols = getattr(BOT, "symbols", None) or ["frxXAUUSD", "frxXAGUSD"]
    intervals = ["5m", "15m", "30m", "1h", "4h"]
    out = {}
    for sym in symbols:
        out[sym] = {}
        for iv in intervals:
            cd = _build_chart_dict(sym, iv, history=history)
            if cd is not None:
                out[sym][iv] = cd
    return out


@app.get("/api/news")
async def api_news():
    return BOT.news_state.to_dict() if BOT else {}


@app.get("/api/session")
async def api_session():
    return BOT.session_info.to_dict() if BOT else {}


@app.get("/api/decisions")
async def api_decisions(symbol: Optional[str] = None, limit: int = 40):
    return BOT.store.recent_decisions(limit, symbol=symbol) if BOT else []


@app.get("/api/trades")
async def api_trades(symbol: Optional[str] = None, limit: int = 40):
    return BOT.store.recent_trades(limit, symbol=symbol) if BOT else []


@app.get("/api/stats")
async def api_stats(symbol: Optional[str] = None):
    return BOT.store.stats(symbol=symbol) if BOT else {}


@app.post("/api/close")
async def api_close_contract(payload: dict = Body(...)):
    """Manually close an active multiplier contract."""
    symbol = payload.get("symbol", "frxXAUUSD")
    if BOT:
        return await BOT.manual_close(symbol)
    return {"ok": False, "error": "Bot not running"}


@app.get("/api/settings")
async def api_settings():
    global CFG
    from .config import Config
    try:
        fresh_cfg = Config.load()
        for k in dir(fresh_cfg):
            if not k.startswith("_") and hasattr(CFG, k):
                setattr(CFG, k, getattr(fresh_cfg, k))
    except Exception as e:
        log.warning("Config reload error in api_settings: %s", e)

    return {
        "fields": settings_schema(CFG),
        "errors": CFG.validate(),
        "restart_pending": _restart_pending()
    }


@app.post("/api/settings")
async def api_settings_save(payload: dict = Body(...)):
    changes = payload.get("changes", payload) or {}
    trial = deepcopy(CFG)
    applied, rejected = {}, {}
    for k, v in changes.items():
        if not hasattr(trial, k):
            rejected[k] = "unknown setting"
            continue
        if k == "token" and v == "***set***":
            continue
        try:
            setattr(trial, k, coerce(trial, k, v))
            applied[k] = getattr(trial, k)
        except Exception as e:
            rejected[k] = str(e)

    errs = trial.validate()
    if errs or rejected:
        return {"ok": False, "errors": errs, "rejected": rejected}

    for k, v in applied.items():
        setattr(CFG, k, v)
    path = CFG.save()
    pending = _restart_pending()

    # Automatically update Deriv client if credentials or trading mode changed
    cred_keys = {"app_id", "token", "mode"}
    changed_creds = [k for k in applied if k in cred_keys]
    if changed_creds and BOT and BOT.client:
        BOT.client.update_credentials(CFG.app_id, CFG.token, CFG.mode)

    # If user enabled validator model, automatically trigger verification & download
    download_started = False
    if applied.get("enable_validator_model") and BOT and hasattr(BOT, "validator"):
        st = BOT.validator.get_status(force_refresh=True)
        if not st.get("is_ready") and st.get("ollama_online"):
            BOT.validator.start_download_async()
            download_started = True

    return {
        "ok": True,
        "saved": path,
        "applied": list(applied),
        "restart_required": [k for k in applied if k in pending and k not in cred_keys],
        "restart_pending": pending,
        "validator_download_started": download_started,
    }


@app.post("/api/deriv/reconnect")
async def api_deriv_reconnect():
    """Forces Deriv WebSocket client to immediately reload credentials and reconnect."""
    if BOT and BOT.client:
        BOT.client.update_credentials(CFG.app_id, CFG.token, CFG.mode)
        return {"ok": True, "message": "Deriv WS reconnect triggered"}
    return {"ok": False, "error": "Bot not active"}


@app.get("/api/validator/status")
async def api_validator_status():
    """Returns current availability, health, and download progress for the validator model."""
    if BOT and hasattr(BOT, "validator"):
        return BOT.validator.get_status(force_refresh=True)
    from .validator_model import ValidatorManager
    return ValidatorManager(CFG).get_status(force_refresh=True)


@app.post("/api/validator/download")
async def api_validator_download():
    """Initiates downloading and caching of the validator model (Llama 3.2 3B)."""
    if BOT and hasattr(BOT, "validator"):
        return BOT.validator.start_download_async()
    from .validator_model import ValidatorManager
    vm = ValidatorManager(CFG)
    return vm.start_download_async()


@app.post("/api/validator/toggle")
async def api_validator_toggle(payload: dict = Body(...)):
    """Quickly toggles the secondary AI validator on/off and starts download if enabled."""
    enabled = payload.get("enabled", not CFG.enable_validator_model)
    CFG.enable_validator_model = bool(enabled)
    CFG.save()
    res = {"ok": True, "enabled": CFG.enable_validator_model}
    if CFG.enable_validator_model:
        vm = BOT.validator if (BOT and hasattr(BOT, "validator")) else None
        if vm:
            st = vm.get_status(force_refresh=True)
            if not st.get("is_ready") and st.get("ollama_online"):
                dl = vm.start_download_async()
                res["download"] = dl
            res["status"] = st
    return res


@app.post("/api/restart")
async def api_restart(force: bool = False):
    global BOT, BOT_TASK, BOOT_CFG
    if BOT_TASK:
        BOT_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await BOT_TASK
    if BOT:
        with contextlib.suppress(Exception):
            await BOT.stop()

    BOT = Bot()
    BOOT_CFG = _snapshot()
    BOT_TASK = asyncio.create_task(BOT.run(), name="bot")
    log.info("Bot restarted with new configuration")
    return {"ok": True}


@app.post("/api/start")
async def api_start():
    if BOT:
        BOT.trading_enabled = True
        BOT.halted_reason = ""
        BOT.eval_trigger.set()
    return {"ok": True, "trading_enabled": True}


@app.post("/api/stop")
async def api_stop():
    if BOT:
        BOT.trading_enabled = False
        BOT.halted_reason = "manual stop"
    return {"ok": True, "trading_enabled": False}


def _page(name: str) -> str:
    with open(os.path.join(TEMPLATES, name), encoding="utf-8") as fh:
        return fh.read()


@app.get("/", response_class=HTMLResponse)
async def index():
    return _page("index.html")
