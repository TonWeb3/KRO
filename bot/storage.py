"""SQLite storage for Deriv Multipliers decisions, trades, and Kronos forecasts."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS multiplier_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    bar_ts          TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    spot            REAL,
    stop_loss       REAL,
    take_profit     REAL,
    stop_loss_usd   REAL,
    take_profit_usd REAL,
    rrr             REAL,
    multiplier      INTEGER,
    stake           REAL,
    conviction      REAL,
    quality_grade   TEXT,
    taken           INTEGER NOT NULL,
    reason          TEXT,
    gates_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_mdec_ts ON multiplier_decisions(ts);

CREATE TABLE IF NOT EXISTS multiplier_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     INTEGER,
    contract_id     INTEGER UNIQUE,
    symbol          TEXT    NOT NULL,
    contract_type   TEXT    NOT NULL,
    timeframe       TEXT,
    opened_ts       REAL    NOT NULL,
    closed_ts       REAL,
    entry_spot      REAL,
    exit_spot       REAL,
    multiplier      INTEGER,
    stake           REAL,
    stop_loss       REAL,
    take_profit     REAL,
    profit          REAL,
    profit_pct      REAL,
    status          TEXT,
    mode            TEXT
);
CREATE INDEX IF NOT EXISTS idx_mtr_opened ON multiplier_trades(opened_ts);

CREATE TABLE IF NOT EXISTS forecasts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    symbol        TEXT    NOT NULL,
    interval      TEXT    NOT NULL,
    bar_ts        TEXT    NOT NULL,
    summary_json  TEXT    NOT NULL,
    mean_path     TEXT,
    band_lo       TEXT,
    band_hi       TEXT,
    y_ts          TEXT
);
CREATE INDEX IF NOT EXISTS idx_fc_ts ON forecasts(ts);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        if not hasattr(self._local, "c"):
            self._local.c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            self._local.c.row_factory = sqlite3.Row
            self._local.c.execute("PRAGMA journal_mode=WAL")
        c = self._local.c
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise

    # ---------------- Writes ----------------

    def log_decision(self, plan, bar_ts: str, taken: bool) -> int:
        d = plan.to_dict()
        with self.conn() as c:
            cur = c.execute(
                """INSERT INTO multiplier_decisions
                   (ts, bar_ts, symbol, direction, timeframe, spot, stop_loss, take_profit,
                    stop_loss_usd, take_profit_usd, rrr, multiplier, stake, conviction,
                    quality_grade, taken, reason, gates_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (time.time(), bar_ts, d["symbol"], d["direction"], d["timeframe"],
                 d["spot"], d["stop_loss"], d["take_profit"], d["stop_loss_usd"],
                 d["take_profit_usd"], d["rrr"], d["multiplier"], d["stake"],
                 d["conviction"], d["quality_grade"], int(taken), d["reason"],
                 json.dumps(d.get("gates", {})))
            )
            return int(cur.lastrowid)

    def open_trade(self, decision_id: int, contract_id: int, plan, buy_price: float,
                   mode: str, entry_spot: Optional[float] = None) -> int:
        now = time.time()
        spot = entry_spot if entry_spot is not None else plan.spot
        with self.conn() as c:
            cur = c.execute(
                """INSERT OR REPLACE INTO multiplier_trades
                   (decision_id, contract_id, symbol, contract_type, timeframe, opened_ts,
                    entry_spot, multiplier, stake, stop_loss, take_profit, status, mode)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (decision_id, contract_id, plan.symbol, plan.direction, plan.timeframe, now,
                 spot, plan.multiplier, buy_price, plan.stop_loss, plan.take_profit,
                 "open", mode)
            )
            return int(cur.lastrowid)

    def close_trade(self, contract_id: int, exit_spot: float, profit: float,
                    status: str) -> None:
        profit_pct = 0.0
        with self.conn() as c:
            row = c.execute("SELECT stake FROM multiplier_trades WHERE contract_id=?", (contract_id,)).fetchone()
            if row and row["stake"] > 0:
                profit_pct = (profit / row["stake"]) * 100.0
            c.execute(
                """UPDATE multiplier_trades SET closed_ts=?, exit_spot=?, profit=?,
                   profit_pct=?, status=? WHERE contract_id=?""",
                (time.time(), exit_spot, profit, profit_pct, status, contract_id)
            )

    def log_forecast(self, fc) -> None:
        lo, hi = fc.band(10, 90)
        with self.conn() as c:
            c.execute(
                """INSERT INTO forecasts
                   (ts, symbol, interval, bar_ts, summary_json, mean_path,
                    band_lo, band_hi, y_ts)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (time.time(), fc.symbol, fc.interval, fc.last_ts.isoformat(),
                 json.dumps(fc.summary()),
                 json.dumps([round(float(x), 6) for x in fc.mean_path]),
                 json.dumps([round(float(x), 6) for x in lo]),
                 json.dumps([round(float(x), 6) for x in hi]),
                 json.dumps([t.isoformat() for t in fc.y_timestamps]))
            )

    # ---------------- Reads ----------------

    def recent_decisions(self, limit: int = 50, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.conn() as c:
            if symbol:
                return [dict(r) for r in c.execute(
                    "SELECT * FROM multiplier_decisions WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit))]
            return [dict(r) for r in c.execute(
                "SELECT * FROM multiplier_decisions ORDER BY id DESC LIMIT ?", (limit,))]

    def recent_trades(self, limit: int = 50, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.conn() as c:
            if symbol:
                return [dict(r) for r in c.execute(
                    "SELECT * FROM multiplier_trades WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit))]
            return [dict(r) for r in c.execute(
                "SELECT * FROM multiplier_trades ORDER BY id DESC LIMIT ?", (limit,))]

    def stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        with self.conn() as c:
            query = "SELECT COUNT(*) n, SUM(profit) pnl, SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) wins, SUM(CASE WHEN profit <= 0 THEN 1 ELSE 0 END) losses FROM multiplier_trades WHERE status != 'open'"
            dec_query = "SELECT COUNT(*) n, SUM(taken) taken FROM multiplier_decisions"
            params = ()
            if symbol:
                query += " AND symbol=?"
                dec_query += " WHERE symbol=?"
                params = (symbol,)
            row = c.execute(query, params).fetchone()
            dec = c.execute(dec_query, params).fetchone()

        n = row["n"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        return {
            "trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / n, 4) if n > 0 else 0.0,
            "profit_usd": round(row["pnl"] or 0.0, 2),
            "decisions": dec["n"] or 0,
            "taken": dec["taken"] or 0,
        }
