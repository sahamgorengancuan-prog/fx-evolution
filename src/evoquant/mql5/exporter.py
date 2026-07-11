"""MQL5 Expert Advisor code generation from a StrategyGenome.

The generated EA mirrors the Tier-A reference semantics:

* signals are evaluated on the just-closed bar (shift 1);
* entries are market orders on the next bar's open tick;
* SL/TP are ATR multiples of the signal bar's ATR;
* sizing = risk%% of equity / stop distance, rounded to the venue lot grid;
* time exit after ``max_bars`` bars; cooldown after every exit;
* simultaneous long+short signals cancel (no trade).

Expressions are transpiled bottom-up: each unique canonical subtree
becomes one MQL5 function named by its semantic hash, so identical
subtrees share code and the transpilation is deterministic — the same
genome always yields byte-identical source.

Signal-dump mode (`InpSignalDump=true`) writes per-bar signal values and
a trade log to CSV in the tester's Files directory; those files feed the
Phase-8 differential parity harness.

Known, documented approximation: the EA's EMA seeds ``EMA_WARMUP_MULT ×
period`` bars back instead of at full history; after the warmup region
the difference is below double precision. Parity is therefore compared
only after the declared warmup (see parity.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from evoquant.data.contracts import InstrumentSpec
from evoquant.errors import EvoquantError
from evoquant.features.ast import Node
from evoquant.features.simplifier import canonicalize
from evoquant.genome.genome import StrategyGenome

EMA_WARMUP_MULT = 10


class ExportError(EvoquantError):
    code = "MQL5_EXPORT"


@dataclass
class _CodeGen:
    functions: dict[str, str]  # func name -> source
    order: list[str]

    def emit(self, name: str, body: str) -> None:
        if name not in self.functions:
            self.functions[name] = body
            self.order.append(name)


_SOURCES = {
    "close": "PxClose",
    "open": "PxOpen",
    "high": "PxHigh",
    "low": "PxLow",
    "volume": "PxVolume",
}

#: window ops: op -> (helper function, width = period + offset)
_WINDOW_OPS = {
    "sma": ("WinMean", 0),
    "highest": ("WinMax", 0),
    "lowest": ("WinMin", 0),
    "stdev": ("WinStd", 0),
    "zscore": ("WinZScore", 0),
    "rank": ("WinRank", 0),
    "delta": ("WinDelta", 1),
    "ret": ("WinRet", 1),
    "efficiency_ratio": ("WinER", 1),
    "rsi": ("WinRSI", 1),
}

_BINARY = {"add": "+", "sub": "-"}
_COMPARE = {"gt": ">", "lt": "<"}


def _fname(node: Node) -> str:
    return f"Expr_{canonicalize(node).structural_hash()[:12]}"


def _gen(node: Node, cg: _CodeGen) -> str:
    """Emit the function for `node` (and children); return its name."""
    name = _fname(node)
    if name in cg.functions:
        return name

    op = node.op
    if op in _SOURCES:
        body = f"double {name}(const int s) {{ return {_SOURCES[op]}(s); }}"
    elif op == "constant":
        body = f"double {name}(const int s) {{ return {node.value!r}; }}"
    elif op == "abs":
        c = _gen(node.children[0], cg)
        body = f"double {name}(const int s) {{ return MathAbs({c}(s)); }}"
    elif op == "neg":
        c = _gen(node.children[0], cg)
        body = f"double {name}(const int s) {{ return -{c}(s); }}"
    elif op == "not":
        c = _gen(node.children[0], cg)
        body = f"double {name}(const int s) {{ return ({c}(s) > 0.5) ? 0.0 : 1.0; }}"
    elif op in _BINARY:
        a = _gen(node.children[0], cg)
        b = _gen(node.children[1], cg)
        body = f"double {name}(const int s) {{ return {a}(s) {_BINARY[op]} {b}(s); }}"
    elif op in _COMPARE:
        a = _gen(node.children[0], cg)
        b = _gen(node.children[1], cg)
        body = (
            f"double {name}(const int s) "
            f"{{ return ({a}(s) {_COMPARE[op]} {b}(s)) ? 1.0 : 0.0; }}"
        )
    elif op in ("and", "or"):
        parts = [f"({_gen(c, cg)}(s) > 0.5)" for c in node.children]
        joiner = " && " if op == "and" else " || "
        body = f"double {name}(const int s) {{ return ({joiner.join(parts)}) ? 1.0 : 0.0; }}"
    elif op in ("cross_above", "cross_below"):
        a = _gen(node.children[0], cg)
        b = _gen(node.children[1], cg)
        if op == "cross_above":
            cond = f"{a}(s+1) <= {b}(s+1) && {a}(s) > {b}(s)"
        else:
            cond = f"{a}(s+1) >= {b}(s+1) && {a}(s) < {b}(s)"
        body = f"double {name}(const int s) {{ return ({cond}) ? 1.0 : 0.0; }}"
    elif op in _WINDOW_OPS:
        helper, offset = _WINDOW_OPS[op]
        c = _gen(node.children[0], cg)
        width = (node.period or 1) + offset
        body = (
            f"double {name}(const int s) {{\n"
            f"   double buf[{width}];\n"
            f"   for(int k = 0; k < {width}; k++) buf[k] = {c}(s + {width} - 1 - k);\n"
            f"   return {helper}(buf, {width});\n"
            f"}}"
        )
    elif op == "ema":
        c = _gen(node.children[0], cg)
        p = node.period or 1
        warm = p * EMA_WARMUP_MULT
        body = (
            f"double {name}(const int s) {{\n"
            f"   double a = 2.0 / ({p} + 1.0);\n"
            f"   double e = {c}(s + {warm});\n"
            f"   for(int j = s + {warm} - 1; j >= s; j--) e = e + a * ({c}(j) - e);\n"
            f"   return e;\n"
            f"}}"
        )
    elif op == "atr":
        p = node.period or 1
        body = f"double {name}(const int s) {{ return AtrSma({p}, s); }}"
    elif op == "chop":
        p = node.period or 2
        body = f"double {name}(const int s) {{ return ChopIdx({p}, s); }}"
    else:
        raise ExportError(f"No MQL5 translation for operator {op!r}", op=op)

    cg.emit(name, body)
    return name


_RUNTIME = r"""
//--- price sources (chronological semantics via shift)
double PxClose(const int s)  { return iClose(_Symbol, _Period, s); }
double PxOpen(const int s)   { return iOpen(_Symbol, _Period, s); }
double PxHigh(const int s)   { return iHigh(_Symbol, _Period, s); }
double PxLow(const int s)    { return iLow(_Symbol, _Period, s); }
double PxVolume(const int s) { return (double)iVolume(_Symbol, _Period, s); }

//--- shared window formulas (mirror evoquant.features.primitives)
double WinMean(const double &b[], const int n) {
   double t = 0; for(int i = 0; i < n; i++) t += b[i]; return t / n;
}
double WinMax(const double &b[], const int n) {
   double m = b[0]; for(int i = 1; i < n; i++) if(b[i] > m) m = b[i]; return m;
}
double WinMin(const double &b[], const int n) {
   double m = b[0]; for(int i = 1; i < n; i++) if(b[i] < m) m = b[i]; return m;
}
double WinStd(const double &b[], const int n) {
   double mu = WinMean(b, n), t = 0;
   for(int i = 0; i < n; i++) t += (b[i] - mu) * (b[i] - mu);
   return MathSqrt(t / n);
}
double WinZScore(const double &b[], const int n) {
   double sd = WinStd(b, n);
   if(sd <= 0) return EMPTY_VALUE;
   return (b[n-1] - WinMean(b, n)) / sd;
}
double WinRank(const double &b[], const int n) {
   int c = 0; for(int i = 0; i < n; i++) if(b[i] <= b[n-1]) c++;
   return (double)c / n;
}
double WinDelta(const double &b[], const int n) { return b[n-1] - b[0]; }
double WinRet(const double &b[], const int n) {
   if(b[0] == 0.0) return EMPTY_VALUE;
   return b[n-1] / b[0] - 1.0;
}
double WinER(const double &b[], const int n) {
   double path = 0;
   for(int i = 1; i < n; i++) path += MathAbs(b[i] - b[i-1]);
   if(path <= 0) return EMPTY_VALUE;
   return MathAbs(b[n-1] - b[0]) / path;
}
double WinRSI(const double &b[], const int n) {
   double up = 0, dn = 0;
   for(int i = 1; i < n; i++) {
      double d = b[i] - b[i-1];
      if(d > 0) up += d; else dn -= d;
   }
   up /= (n - 1); dn /= (n - 1);
   if(up + dn == 0.0) return 50.0;
   return 100.0 * up / (up + dn);
}
double TrueRangeAt(const int s) {
   double h = PxHigh(s), l = PxLow(s), pc = PxClose(s + 1);
   if(pc == 0.0 || s + 1 >= Bars(_Symbol, _Period)) return h - l;
   return MathMax(h - l, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
}
double AtrSma(const int p, const int s) {
   double t = 0; for(int k = 0; k < p; k++) t += TrueRangeAt(s + k); return t / p;
}
double ChopIdx(const int p, const int s) {
   double tr = 0, hh = PxHigh(s), ll = PxLow(s);
   for(int k = 0; k < p; k++) {
      tr += TrueRangeAt(s + k);
      double h = PxHigh(s + k), l = PxLow(s + k);
      if(h > hh) hh = h;
      if(l < ll) ll = l;
   }
   double rng = hh - ll;
   if(rng <= 0 || tr <= 0) return EMPTY_VALUE;
   return MathLog10(tr / rng) / MathLog10((double)p);
}
"""

_EA_TEMPLATE = """//+------------------------------------------------------------------+
//| {ea_name}.mq5                                                     |
//| GENERATED by evoquant {version} — DO NOT EDIT BY HAND             |
//| genome_hash: {genome_hash}                                        |
//| symbol: {symbol}  (validate with: Every tick based on real ticks) |
//+------------------------------------------------------------------+
#property strict

input bool   InpSignalDump   = false; // dump per-bar signals + trades to CSV
input double InpRiskPerTrade = {risk_per_trade}; // fraction of equity per trade
input int    InpMagic        = 992601;

// risk block (from genome)
const double SL_ATR   = {sl_atr};
const double TP_ATR   = {tp_atr};
const int    MAX_BARS = {max_bars};
const int    COOLDOWN = {cooldown_bars};
const int    MIN_BARS = {min_bars}; // total causal lookback + margin

// venue contract (from instrument spec — verify against broker symbol!)
const double LOT_STEP = {lot_step};
const double MIN_LOT  = {min_lot};
const double MAX_LOT  = {max_lot};

{runtime}

//--- generated expression functions ---------------------------------
{expressions}

double SigEntryLong(const int s)  {{ return {entry_long_fn}(s); }}
double SigEntryShort(const int s) {{ return {entry_short_fn}(s); }}
double SigAllow(const int s)      {{ return {allow_fn}(s); }}
double SigAtr(const int s)        {{ return {atr_fn}(s); }}

//--- state -----------------------------------------------------------
datetime g_last_bar = 0;
int      g_cooldown_until_bar = -1;
int      g_bar_index = 0;
int      g_entry_bar = -1;
int      g_pending_dir = 0;
double   g_pending_atr = 0.0;
int      g_sig_file = INVALID_HANDLE;
int      g_trade_file = INVALID_HANDLE;

int OnInit() {{
   if(InpSignalDump) {{
      g_sig_file = FileOpen("evoquant_signals.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ';');
      FileWrite(g_sig_file, "time", "entry_long", "entry_short", "allow", "atr");
      g_trade_file = FileOpen("evoquant_trades.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ';');
      FileWrite(g_trade_file, "open_time", "close_time", "direction", "lots",
                "entry_price", "exit_price", "sl", "tp", "exit_reason");
   }}
   return INIT_SUCCEEDED;
}}

void OnDeinit(const int reason) {{
   if(g_sig_file != INVALID_HANDLE) FileClose(g_sig_file);
   if(g_trade_file != INVALID_HANDLE) FileClose(g_trade_file);
}}

double RoundLots(double lots) {{
   if(lots < MIN_LOT) return 0.0;
   double steps = MathFloor((lots - MIN_LOT) / LOT_STEP + 1e-9);
   return MathMin(MIN_LOT + steps * LOT_STEP, MAX_LOT);
}}

// NOTE: order management (open/close/SL/TP/max-bars/cooldown) mirrors the
// Python Tier-A engine bar-for-bar; the tick path between bars is resolved
// by the Strategy Tester's real ticks — that difference IS the experiment.
void OnTick() {{
   datetime bar_time = iTime(_Symbol, _Period, 0);
   if(bar_time == g_last_bar) {{ ManagePosition(); return; }}
   g_last_bar = bar_time;
   g_bar_index++;

   // 1. execute the pending entry decided on the previous closed bar
   if(g_pending_dir != 0 && !PositionSelect(_Symbol) && g_bar_index > g_cooldown_until_bar)
      OpenPosition(g_pending_dir, g_pending_atr);
   g_pending_dir = 0;

   if(Bars(_Symbol, _Period) < MIN_BARS) return;

   // 2. evaluate signals on the just-closed bar (shift 1)
   double eL = SigEntryLong(1), eS = SigEntryShort(1);
   double allow = SigAllow(1), atr = SigAtr(1);
   if(InpSignalDump && g_sig_file != INVALID_HANDLE)
      FileWrite(g_sig_file, (long)iTime(_Symbol, _Period, 1),
                (int)(eL > 0.5), (int)(eS > 0.5), (int)(allow > 0.5),
                DoubleToString(atr, 10));

   if(PositionSelect(_Symbol) || allow <= 0.5 || g_bar_index < g_cooldown_until_bar)
      return;
   bool wantL = eL > 0.5, wantS = eS > 0.5;
   if(wantL == wantS) return; // conflict or nothing => no trade
   if(atr == EMPTY_VALUE || atr <= 0.0) return;
   g_pending_dir = wantL ? 1 : -1;
   g_pending_atr = atr;
}}

void OpenPosition(const int dir, const double atr) {{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double stop_dist = SL_ATR * atr;
   if(stop_dist <= 0) return;
   double qty = equity * InpRiskPerTrade / stop_dist;
   double lots = RoundLots(qty / SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE));
   if(lots <= 0.0) return;
   MqlTradeRequest req; MqlTradeResult res;
   ZeroMemory(req); ZeroMemory(res);
   req.action = TRADE_ACTION_DEAL;
   req.symbol = _Symbol;
   req.volume = lots;
   req.magic = InpMagic;
   req.deviation = 10;
   if(dir > 0) {{
      req.type = ORDER_TYPE_BUY;
      req.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      req.sl = req.price - stop_dist;
      req.tp = req.price + TP_ATR * atr;
   }} else {{
      req.type = ORDER_TYPE_SELL;
      req.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      req.sl = req.price + stop_dist;
      req.tp = req.price - TP_ATR * atr;
   }}
   if(OrderSend(req, res)) g_entry_bar = g_bar_index;
}}

void ManagePosition() {{
   if(!PositionSelect(_Symbol)) return;
   if(g_bar_index - g_entry_bar >= MAX_BARS) ClosePosition("max_bars");
}}

void ClosePosition(const string reason) {{
   MqlTradeRequest req; MqlTradeResult res;
   ZeroMemory(req); ZeroMemory(res);
   long ptype = PositionGetInteger(POSITION_TYPE);
   req.action = TRADE_ACTION_DEAL;
   req.symbol = _Symbol;
   req.volume = PositionGetDouble(POSITION_VOLUME);
   req.magic = InpMagic;
   req.deviation = 10;
   req.type = (ptype == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.price = (ptype == POSITION_TYPE_BUY)
               ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
               : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(OrderSend(req, res)) {{
      g_cooldown_until_bar = g_bar_index + COOLDOWN;
      if(InpSignalDump && g_trade_file != INVALID_HANDLE)
         FileWrite(g_trade_file, (long)PositionGetInteger(POSITION_TIME),
                   (long)TimeCurrent(),
                   (ptype == POSITION_TYPE_BUY) ? 1 : -1,
                   DoubleToString(req.volume, 2),
                   DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN), 10),
                   DoubleToString(req.price, 10),
                   DoubleToString(PositionGetDouble(POSITION_SL), 10),
                   DoubleToString(PositionGetDouble(POSITION_TP), 10),
                   reason);
   }}
}}
"""


def export_ea(genome: StrategyGenome, spec: InstrumentSpec, min_bars_margin: int = 50) -> str:
    """Transpile a genome into deterministic MQL5 EA source."""
    from evoquant import __version__
    from evoquant.features.compiler import compile_feature

    cg = _CodeGen(functions={}, order=[])
    entry_long = canonicalize(genome.entry_long)
    entry_short = canonicalize(genome.entry_short)
    allow = canonicalize(genome.allow)
    atr_node = Node(op="atr", period=genome.risk.atr_period)

    long_fn = _gen(entry_long, cg)
    short_fn = _gen(entry_short, cg)
    allow_fn = _gen(allow, cg)
    atr_fn = _gen(atr_node, cg)

    lookback = max(
        compile_feature(entry_long).lookback,
        compile_feature(entry_short).lookback,
        compile_feature(allow).lookback,
        compile_feature(atr_node).lookback,
    )
    # EMA warmup horizons extend the required history
    ema_extra = max(
        ((n.period or 1) * EMA_WARMUP_MULT for tree in (entry_long, entry_short, allow)
         for n in tree.walk() if n.op == "ema"),
        default=0,
    )
    min_bars = lookback + ema_extra + min_bars_margin

    expressions = "\n\n".join(cg.functions[name] for name in cg.order)
    return _EA_TEMPLATE.format(
        ea_name=f"Evoquant_{genome.symbol}_{genome.genome_hash()[:10]}",
        version=__version__,
        genome_hash=genome.genome_hash(),
        symbol=genome.symbol,
        risk_per_trade=genome.risk.risk_per_trade,
        sl_atr=genome.risk.sl_atr,
        tp_atr=genome.risk.tp_atr,
        max_bars=genome.risk.max_bars,
        cooldown_bars=genome.risk.cooldown_bars,
        min_bars=min_bars,
        lot_step=spec.lot_step,
        min_lot=spec.min_lot,
        max_lot=spec.max_lot,
        runtime=_RUNTIME,
        expressions=expressions,
        entry_long_fn=long_fn,
        entry_short_fn=short_fn,
        allow_fn=allow_fn,
        atr_fn=atr_fn,
    )


def strategy_tester_ini(
    symbol: str,
    ea_name: str,
    from_date: str,
    to_date: str,
    deposit: float,
    timeframe: str = "H1",
) -> str:
    """MetaTrader 5 Strategy Tester .ini — Model=4 is 'Every tick based on
    real ticks'; anything else must not be called tick validation."""
    return (
        "[Tester]\n"
        f"Expert=Experts\\evoquant\\{ea_name}\n"
        f"Symbol={symbol}\n"
        f"Period={timeframe}\n"
        "Model=4\n"  # Every tick based on real ticks — the whole point
        f"FromDate={from_date}\n"
        f"ToDate={to_date}\n"
        f"Deposit={deposit:.0f}\n"
        "Leverage=1\n"
        "ExecutionMode=1\n"
        "Optimization=0\n"
        "Visual=0\n"
        "ShutdownTerminal=1\n"
        "Report=evoquant_report\n"
        "ReplaceReport=1\n"
    )


RUN_INSTRUCTIONS = """# Running the real-tick validation (manual — no terminal in CI)

Parity status is PENDING_REAL_TICK until these steps are executed and the
resulting CSVs are fed back to `evoquant.mql5.parity`.

1. If the broker does not provide {symbol}: create a Custom Symbol
   (View > Symbols > Create Custom Symbol) using the archived contract
   spec in `instrument_spec.json`, then import real ticks
   (Symbols > {symbol} > Ticks > Import).
2. Copy `{ea_name}.mq5` to `MQL5/Experts/evoquant/` and compile in
   MetaEditor (F7). Zero warnings expected.
3. Run the Strategy Tester with the provided config:
       terminal64.exe /config:tester.ini
   Model MUST remain "Every tick based on real ticks" (Model=4).
4. Re-run once with `InpSignalDump=true`; collect from the tester's
   Files directory:
       evoquant_signals.csv
       evoquant_trades.csv
5. Feed them back:
       python -m evoquant.cli mql5-parity \\
           --bundle <this directory> --signals evoquant_signals.csv \\
           --trades evoquant_trades.csv
   The parity report itemizes every discrepancy; discrepancies are
   release blockers, not footnotes.
"""
