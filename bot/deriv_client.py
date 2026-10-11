"""Deriv WebSocket Client for Multi-Asset Multipliers (Gold & Silver).

Handles:
- Real-time tick streaming for both frxXAUUSD (Gold) and frxXAGUSD (Silver)
- Multi-timeframe historical candlestick retrieval (ticks_history)
- Deriv Multiplier proposal pricing with attached Stop Loss ($) and Take Profit ($)
- Contract purchase execution, manual exit ('sell'), and real-time open position tracking
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

log = logging.getLogger(__name__)

REST_BASE = "https://api.derivws.com/trading/v1"
WS_FALLBACK = "wss://ws.derivws.com/websockets/v3"

def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class DerivError(Exception):
    def __init__(self, code: Any, message: str, subcode: Optional[str] = None, code_args: Optional[list] = None):
        self.code = code
        self.message = message
        self.subcode = subcode
        self.code_args = code_args or []
        super().__init__(f"[{code}] {message}")


class DerivClient:
    def __init__(self, app_id: str, token: str, symbols: Optional[List[str]] = None,
                 mode: str = "demo", rest_base: str = REST_BASE,
                 on_connect: Optional[Callable] = None):
        self.app_id = str(app_id or "1089")
        self.token = token or ""
        self.symbols = symbols or ["frxXAUUSD", "frxXAGUSD"]
        self.mode = (mode or "demo").lower()
        self.rest_base = rest_base
        self.on_connect = on_connect

        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.closed = False
        self.connected = False
        self.authorized = False

        # Multi-symbol price tracking
        self.prices: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.last_ts: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.pip_sizes: Dict[str, int] = {s: 2 for s in self.symbols}

        self.balance: Optional[float] = None
        self.currency: str = "USD"
        self.accounts: List[Dict[str, Any]] = []
        self.account_id: Optional[str] = None
        self.account_type: Optional[str] = None
        self.last_error: Optional[str] = None

        self.on_ohlc: Optional[Callable[[Dict[str, Any]], None]] = None

        self._req_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def update_credentials(self, app_id: str, token: str, mode: str = "demo") -> None:
        """Dynamically update credentials and reconnect without restarting entire process."""
        self.app_id = str(app_id or "1089")
        self.token = token or ""
        self.mode = (mode or "demo").lower()
        self.accounts = []
        log.info("Deriv credentials dynamically updated: app_id=%s, mode=%s (triggering WS reconnect)", self.app_id, self.mode)
        if self.ws and not self.ws.closed:
            asyncio.create_task(self.ws.close())

    @property
    def is_virtual(self) -> bool:
        return self.account_type == "demo" if self.account_type else (self.mode == "demo")

    def get_spot(self, symbol: str) -> Optional[float]:
        p = self.prices.get(symbol, 0.0)
        return p if p > 0 else None

    def _headers(self) -> Dict[str, str]:
        hdrs = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        if self.app_id:
            hdrs["Deriv-App-ID"] = str(self.app_id)
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
        return hdrs

    def _rest_sync(self, method: str, path: str, json_data: Optional[Dict] = None, timeout: float = 10.0) -> Dict[str, Any]:
        import requests
        url = self.rest_base + path
        hdrs = self._headers()
        resp = requests.request(method, url, headers=hdrs, json=json_data if json_data is not None else {}, timeout=timeout)
        if resp.status_code >= 400:
            raise DerivError(resp.status_code, resp.text[:120])
        try:
            return resp.json()
        except Exception:
            return {"_raw": resp.text}

    async def _rest(self, method: str, path: str, json_data: Optional[Dict] = None, timeout: float = 10.0) -> Dict[str, Any]:
        return await asyncio.to_thread(self._rest_sync, method, path, json_data, timeout)

    async def fetch_accounts(self) -> List[Dict[str, Any]]:
        try:
            data = await self._rest("GET", "/options/accounts")
            self.accounts = data.get("data") or []
            return self.accounts
        except Exception as e:
            log.warning("REST accounts fetch failed: %s", e)
            return []

    async def _get_otp_ws_url(self, account_id: str) -> Optional[str]:
        try:
            data = await self._rest("POST", f"/options/accounts/{account_id}/otp")
            return (data.get("data") or {}).get("url")
        except Exception as e:
            log.warning("OTP fetch failed: %s", e)
            return None

    def _select_account(self) -> Optional[Dict[str, Any]]:
        want = "demo" if self.mode == "demo" else "real"
        for a in self.accounts:
            if (a.get("account_type") or "").lower() == want:
                return a
        return self.accounts[0] if self.accounts else None

    async def start(self):
        """Main connection and message routing loop."""
        self._loop = asyncio.get_running_loop()
        while not self.closed:
            try:
                ws_url = None
                use_otp = False

                if self.token:
                    await self.fetch_accounts()
                    acct = self._select_account()
                    if acct:
                        self.account_id = acct.get("account_id")
                        self.account_type = (acct.get("account_type") or "").lower()
                        if acct.get("balance") is not None:
                            self.balance = float(acct["balance"])
                        self.currency = acct.get("currency", self.currency)
                        ws_url = await self._get_otp_ws_url(self.account_id)
                        if ws_url:
                            use_otp = True

                if not ws_url:
                    # For direct WS fallback to ws.derivws.com, app_id MUST be numeric (default 1089)
                    # Non-numeric IDs (e.g. PAT app IDs) will cause Cloudflare 520 errors on ws.derivws.com
                    fallback_id = self.app_id if (self.app_id and self.app_id.isdigit()) else "1089"
                    ws_url = f"{WS_FALLBACK}?app_id={fallback_id}"

                client_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                if self.app_id and str(self.app_id).isdigit():
                    client_headers["Deriv-App-ID"] = str(self.app_id)

                self._session = aiohttp.ClientSession(headers=client_headers)
                async with self._session.ws_connect(ws_url, heartbeat=20) as ws:
                    self.ws = ws
                    self.connected = True
                    self.last_error = None
                    log.info("Deriv Multiplier WS connected (%s, mode=%s, acct=%s)",
                             "OTP" if use_otp else "direct", self.mode, self.account_id or "unauth")

                    if not use_otp and self.token:
                        auth_resp = await self._send({"authorize": self.token})
                        if "authorize" in auth_resp:
                            a = auth_resp["authorize"]
                            self.authorized = True
                            self.account_id = a.get("loginid")
                            self.account_type = "demo" if a.get("is_virtual") else "real"
                            self.balance = _f(a.get("balance"))
                            self.currency = a.get("currency", self.currency)
                            log.info("Authorized as %s (%s) balance=%s %s",
                                     self.account_id, self.account_type, self.balance, self.currency)
                    else:
                        self.authorized = True if use_otp else False

                    # Subscribe to ticks for all metals (Gold and Silver)
                    for sym in self.symbols:
                        await self._send({"ticks": sym, "subscribe": 1}, wait=False)
                    await self._send({"balance": 1, "subscribe": 1}, wait=False)
                    asyncio.create_task(self.fetch_symbols_info())
                    asyncio.create_task(self.fetch_balance())

                    if self.on_connect:
                        try:
                            cb_res = self.on_connect()
                            if asyncio.iscoroutine(cb_res):
                                asyncio.create_task(cb_res)
                        except Exception as e:
                            log.warning("Deriv on_connect callback exception: %s", e)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._dispatch(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("Deriv connection error: %s. Reconnecting in 3s...", e)
            finally:
                self.connected = False
                self.authorized = False
                self.ws = None
                if self._session:
                    await self._session.close()
                    self._session = None
                for fut in list(self._pending.values()):
                    if not fut.done():
                        fut.set_exception(ConnectionError("Deriv WS dropped"))
                self._pending.clear()

            if not self.closed:
                await asyncio.sleep(3)

    def close(self):
        self.closed = True
        if self.ws:
            asyncio.create_task(self.ws.close())

    async def _send(self, payload: Dict[str, Any], wait: bool = True, timeout: float = 15.0) -> Dict[str, Any]:
        if self.ws is None or self.ws.closed:
            raise ConnectionError("Deriv WS not connected")
        self._req_id += 1
        rid = self._req_id
        payload = dict(payload, req_id=rid)
        if wait:
            fut = self._loop.create_future()
            self._pending[rid] = fut
        await self.ws.send_str(json.dumps(payload))
        if not wait:
            return {}
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(rid, None)

    def _dispatch(self, data: Dict[str, Any]):
        rid = data.get("req_id")
        mtype = data.get("msg_type")

        if mtype == "tick" and "tick" in data:
            t = data["tick"]
            sym = t.get("symbol")
            q = t.get("quote")
            if sym and q is not None:
                self.prices[sym] = float(q)
                self.last_ts[sym] = float(t.get("epoch") or time.time())
                if t.get("pip_size") is not None:
                    try:
                        self.pip_sizes[sym] = int(t["pip_size"])
                    except Exception:
                        pass

        elif mtype == "balance" and "balance" in data:
            b = data["balance"]
            v = _f(b.get("balance"))
            if v is not None:
                self.balance = v
                self.currency = b.get("currency", self.currency)

        elif mtype == "ohlc" and "ohlc" in data:
            ohlc = data["ohlc"]
            sym = ohlc.get("symbol")
            if sym and ohlc.get("close") is not None:
                try:
                    self.prices[sym] = float(ohlc["close"])
                    self.last_ts[sym] = float(ohlc.get("epoch") or time.time())
                except Exception:
                    pass
            if self.on_ohlc:
                try:
                    res = self.on_ohlc(ohlc)
                    if asyncio.iscoroutine(res):
                        asyncio.create_task(res)
                except Exception as e:
                    log.debug("on_ohlc error: %s", e)

        if "error" in data:
            err = data["error"] or {}
            self.last_error = f"{err.get('code')}: {err.get('message')}"

        fut = self._pending.get(rid) if rid is not None else None
        if fut is not None and not fut.done():
            if "error" in data:
                err = data["error"] or {}
                fut.set_exception(DerivError(err.get("code"), err.get("message"),
                                             err.get("subcode"), err.get("code_args")))
            else:
                fut.set_result(data)

    async def fetch_symbols_info(self) -> None:
        try:
            resp = await self._send({"active_symbols": "brief"})
            for x in (resp.get("active_symbols") or []):
                sym = x.get("symbol") or x.get("underlying_symbol")
                if sym in self.symbols:
                    pv = x.get("pip_size") if x.get("pip_size") is not None else x.get("pip")
                    if isinstance(pv, (int, float)) and 0 < pv < 1:
                        self.pip_sizes[sym] = max(0, round(-math.log10(pv)))
                    elif pv is not None:
                        self.pip_sizes[sym] = int(pv)
        except Exception as e:
            log.warning("fetch_symbols_info error: %s", e)

    async def fetch_balance(self) -> Optional[float]:
        try:
            resp = await self._send({"balance": 1})
            b = resp.get("balance") or {}
            v = _f(b.get("balance"))
            if v is not None:
                self.balance = v
                self.currency = b.get("currency", self.currency)
            return v
        except Exception as e:
            log.warning("fetch_balance error: %s", e)
            return self.balance

    async def fetch_candles(self, symbol: str, granularity: int = 300, count: int = 2000) -> List[Dict[str, Any]]:
        """Fetch historical closed candles via ticks_history for any symbol and timeframe,
        paginating backwards if necessary to reach target count (up to 2000+ bars).
        """
        try:
            span_ms = granularity * 1000
            accumulated: List[Dict[str, Any]] = []
            current_end: Any = "latest"
            batch_size = min(count, 1000)

            while len(accumulated) < count:
                needed = min(batch_size, count - len(accumulated))
                payload = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": granularity,
                    "count": needed,
                    "end": current_end
                }
                resp = await self._send(payload, timeout=25.0)
                candles = resp.get("candles", [])
                if not candles:
                    break

                batch_out = []
                for c in candles:
                    epoch_ms = int(c["epoch"]) * 1000
                    batch_out.append({
                        "open_time": epoch_ms,
                        "close_time": epoch_ms + span_ms - 1,
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": float(c.get("volume", 0.0) or 0.0),
                        "amount": float(c.get("amount", 0.0) or 0.0)
                    })

                if current_end == "latest":
                    accumulated = batch_out
                else:
                    existing_times = {x["open_time"] for x in accumulated}
                    new_unique = [x for x in batch_out if x["open_time"] not in existing_times]
                    if not new_unique:
                        break
                    accumulated = new_unique + accumulated

                if len(accumulated) >= count or len(candles) < 50:
                    break

                oldest_epoch = int(accumulated[0]["open_time"] // 1000)
                current_end = oldest_epoch - 1

            accumulated.sort(key=lambda x: x["open_time"])
            return accumulated[-count:]
        except Exception as e:
            self.last_error = f"fetch_candles ({symbol}): {e}"
            log.error("Failed to fetch candles for %s (g=%d): %s", symbol, granularity, e)
            return []

    async def subscribe_candles_stream(self, symbol: str, granularity: int = 300) -> bool:
        """Subscribes to live streaming OHLC updates on WebSocket for a symbol and granularity."""
        try:
            await self._send({
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": 1,
                "end": "latest",
                "subscribe": 1
            }, wait=False)
            return True
        except Exception as e:
            log.warning("Failed to subscribe candles stream for %s (g=%d): %s", symbol, granularity, e)
            return False

    # ---------------- Deriv Multipliers Trading Methods ----------------

    async def get_multiplier_proposal(self, symbol: str, contract_type: str, amount: float,
                                      multiplier: int, stop_loss_usd: float,
                                      take_profit_usd: Optional[float] = None,
                                      cancellation: Optional[str] = None) -> Dict[str, Any]:
        """Request a Multiplier contract proposal with attached Stop Loss ($) and Take Profit ($)."""
        amt = round(float(amount), 2)
        # In Deriv Multipliers, stop_loss cannot exceed stake (maximum loss is 100% of stake).
        # Enforce that SL is capped well below the stake (preserving capital and preventing 100% stake death).
        max_sl_cap = min(round(amt * 0.70, 2), max(0.50, round(amt - 0.50, 2)))
        safe_sl = round(min(float(stop_loss_usd), max_sl_cap), 2)
        if safe_sl < 0.50 and amt >= 1.0:
            safe_sl = 0.50

        req = {
            "proposal": 1,
            "amount": amt,
            "basis": "stake",
            "contract_type": contract_type.upper(),  # 'MULTUP' or 'MULTDOWN'
            "currency": self.currency or "USD",
            "underlying_symbol": symbol,
            "multiplier": int(multiplier),
            "limit_order": {
                "stop_loss": safe_sl
            }
        }
        if take_profit_usd and take_profit_usd > 0:
            req["limit_order"]["take_profit"] = round(float(take_profit_usd), 2)
        if cancellation:
            req["cancellation"] = str(cancellation)

        try:
            resp = await self._send(req)
            p = resp.get("proposal", {})
            return {
                "ok": True,
                "id": p.get("id"),
                "ask_price": _f(p.get("ask_price")),
                "spot": _f(p.get("spot")),
                "spot_time": p.get("spot_time"),
                "multiplier": multiplier,
                "stop_loss_usd": stop_loss_usd,
                "take_profit_usd": take_profit_usd,
            }
        except DerivError as e:
            return {"ok": False, "error": f"{e.code}: {e.message}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def buy(self, proposal_id: str, price: float) -> Dict[str, Any]:
        """Purchase the priced multiplier contract."""
        try:
            resp = await self._send({"buy": proposal_id, "price": round(float(price), 2)})
            b = resp.get("buy", {})
            if b.get("balance_after") is not None:
                self.balance = _f(b["balance_after"])
            return {
                "ok": True,
                "contract_id": b.get("contract_id"),
                "buy_price": _f(b.get("buy_price")),
                "shortcode": b.get("shortcode")
            }
        except DerivError as e:
            return {"ok": False, "error": f"{e.code}: {e.message}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def sell(self, contract_id: int) -> Dict[str, Any]:
        """Manually close an active multiplier position early to take profit or cut loss."""
        try:
            resp = await self._send({"sell": contract_id, "price": 0})
            s = resp.get("sell", {})
            return {
                "ok": True,
                "contract_id": s.get("contract_id"),
                "sold_for": _f(s.get("sold_for")),
            }
        except DerivError as e:
            return {"ok": False, "error": f"{e.code}: {e.message}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def contract_status(self, contract_id: int) -> Dict[str, Any]:
        """Fetch real-time floating status of an open multiplier position."""
        try:
            resp = await self._send({"proposal_open_contract": 1, "contract_id": contract_id})
            c = resp.get("proposal_open_contract", {})
            lo = c.get("limit_order", {})
            sl_val = lo.get("stop_loss", {})
            tp_val = lo.get("take_profit", {})
            sl_amt = sl_val.get("order_amount") if isinstance(sl_val, dict) else (sl_val.get("value") if isinstance(sl_val, dict) else sl_val)
            tp_amt = tp_val.get("order_amount") if isinstance(tp_val, dict) else (tp_val.get("value") if isinstance(tp_val, dict) else tp_val)

            return {
                "ok": True,
                "contract_id": c.get("contract_id"),
                "is_sold": bool(c.get("is_sold")),
                "is_expired": bool(c.get("is_expired")),
                "status": c.get("status"),          # 'open', 'won', 'lost'
                "profit": _f(c.get("profit")),
                "profit_percentage": _f(c.get("profit_percentage")),
                "buy_price": _f(c.get("buy_price")),
                "current_spot": _f(c.get("current_spot")),
                "entry_spot": _f(c.get("entry_spot")),
                "multiplier": c.get("multiplier"),
                "stop_loss": _f(sl_amt),
                "take_profit": _f(tp_amt),
                "date_start": c.get("date_start"),
                "date_expiry": c.get("date_expiry"),
            }
        except DerivError as e:
            return {"ok": False, "error": f"{e.code}: {e.message}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
