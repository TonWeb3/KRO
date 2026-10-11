# Kronos Deriv — Gold & Silver Multiplier Autonomous Trading System

An institutional-grade autonomous **Deriv Multipliers (`MULTUP` / `MULTDOWN`)** trading system driven by [Kronos](https://github.com/shiyu-coder/Kronos), the foundation model for financial candlesticks.

Engineered specifically for **Gold (`frxXAUUSD`) and Silver (`frxXAGUSD`)**, the bot leverages multi-timeframe analysis (5m, 15m, 30m, and 1h macro), algorithmic Support & Resistance detection, adaptive balance sizing, global session clock liquidity, and a high-impact US macroeconomic news breakout strategy.

---

## 1. Core Architecture & Strategy Workflow

```
Market Data (Gold & Silver Ticks + Candles: 5m, 15m, 30m, 1h)
   │
   ├─► Algorithmic Support & Resistance Detection:
   │      • Detects swing high/low fractals and volume-weighted price clusters.
   │      • Maps zones: AT_SUPPORT, AT_RESISTANCE, or MID_RANGE.
   │
   ├─► Balance-Adaptive Timeframe Sizing:
   │      • Low Balance (< $250): Forces 5m execution for micro-pip SL distance.
   │      • Mid Balance ($250–$750): Unlocks 15m execution.
   │      • High Balance (> $750): Unlocks 30m execution.
   │
   ├─► Global Session Clock & Liquidity Filter:
   │      • Identifies London, New York, Tokyo, and Sydney trading sessions.
   │      • Highlights the London / NY Golden Overlap (12:00–16:00 UTC).
   │
   ├─► High-Impact Macro Economic News Engine:
   │      • Tracks US CPI, NFP, FOMC, PPI, Retail Sales releases.
   │      • Activates News Breakout Momentum Strategy on post-news volume expansion.
   │
   ├─► Kronos Foundation Model Monte Carlo Inference:
   │      • Generates 24 forward trajectories across future bars.
   │      • Evaluates directional conviction and volatility expansion bounds.
   │
   ├─► SL-Based Multiplier Optimizer:
   │      • Calculates: Stake × Multiplier = Target Risk ($) / SL Distance (%).
   │      • Dynamically selects optimal Multiplier (20x, 40x, 50x, 100x, 200x).
   │      • Attaches Deriv-native hard Stop Loss ($) to guarantee equity is never wiped out.
   │
   └─► If Autonomous Trading is ACTIVE:
          • Submits Deriv Multiplier proposal (`MULTUP` or `MULTDOWN`).
          • Buys contract and streams real-time floating P&L, current spot, and SL/TP.
```

---

## 2. Key Features

- **Concurrent Dual-Asset Trading**: Trades Gold (`frxXAUUSD`) and Silver (`frxXAGUSD`) simultaneously with cross-asset correlation confirmation.
- **Support & Resistance Precision**: Enters Long only near Support with tight SL beneath it; enters Short only near Resistance with tight SL above it.
- **Dynamic Multiplier Protection**: Never trade fixed arbitrary leverage. Multipliers ($20\times$ to $200\times$) and stakes are sized so hitting the tight SL equals your exact risk budget (e.g., 1.5% of balance).
- **Balance-Adaptive Engine**: Smaller accounts automatically stick to lower timeframes (5m) to capture high-leverage moves with minimal dollar drawdown.
- **Macro News Breakout Engine**: Leverages high-impact news releases (CPI, NFP, FOMC) to capture massive post-news momentum runs using Kronos probability validation.
- **Cyber-Teal Real-Time Dashboard**:
  - Gold & Silver asset switcher tabs.
  - Interactive SVG Candlestick chart with overlaid S/R levels and Monte Carlo uncertainty envelopes.
  - Global Session Clock with live UTC time and Golden Overlap indicator.
  - High-Impact Macro News calendar ticker and countdown.
  - Real-time Active Multiplier Contracts card with live floating P&L and manual close button.
  - Full audit trail of decisions and settled trades stored in SQLite.

---

## 3. Quick Start

### Installation
```bash
git clone <your-repo-url>
cd "Kronos Deriv - Multiplier"
pip install -r requirements.txt
```

### Verification
```bash
python main.py --check
```

### Launch Bot & Dashboard
```bash
python main.py
```
Open browser to `http://localhost:8120`. Click the green **`▶ Start`** button to begin autonomous trading.
