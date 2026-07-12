"""Offline Gradio one-click UI for evoquant.

Wraps the pair-agnostic orchestrator in a local Gradio app that opens in
the browser. Fully offline: no external services, no Cloudflare. Any
ForexSB JSON in the data dir becomes a runnable pair (BNBUSDT is only the
committed sample). Gradio is an optional dependency — importing this
module without it raises a clear message.

Design: the orchestrator's `on_progress` callback pushes events into a
queue; the Gradio run handler is a generator that drains the queue and
streams live status, a telemetry table, verdict, metrics, the full report
and a downloadable file.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from evoquant.experiment.orchestrator import RunConfig, run_experiment
from evoquant.webui.server import discover_pairs

try:  # optional dependency; the CLI/.bat installs it
    import gradio as gr
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the .bat
    raise ModuleNotFoundError(
        "gradio is not installed. Run:  pip install gradio   "
        "(the one-click run_all.bat / run_all.sh does this for you)."
    ) from exc


DISCLAIMER = (
    "All numbers come from the **BAR_APPROXIMATION** reference engine under "
    "**explicit, user-set cost assumptions**. Tick/event and MQL5 real-tick "
    "validation are PENDING for every candidate. `NO_EDGE_FOUND` is a normal, "
    "honest outcome — a pass is never forced, and the final lockbox is never "
    "opened by a run."
)


def _pairs_table(data_dir: str) -> tuple[list[list[Any]], dict[str, str]]:
    pairs = discover_pairs(data_dir)
    rows = [[p["symbol"], p["name"], p["n_bars"], f"M{p['period_minutes']}"] for p in pairs]
    label_to_file = {f"{p['symbol']}  ·  {p['name']}": p["file"] for p in pairs}
    return rows, label_to_file


def _telemetry_rows(events: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for e in events:
        if e.get("stage") == "search_generation":
            rows.append([
                e.get("fold"),
                e.get("generation"),
                round(float(e.get("hypervolume", 0.0)), 5),
                round(float(e.get("feasibility_rate", 0.0)), 3),
                e.get("archive_coverage"),
                e.get("intervention") or "",
            ])
    return rows


def _metrics_rows(report: dict[str, Any]) -> list[list[Any]]:
    champ = report.get("champion", {}) or {}
    oos = champ.get("oos_eval", {}) or {}
    val = report.get("validation", {}) or {}
    dsr = val.get("deflated_sharpe", {})
    dsr_v = dsr.get("deflated_sharpe") if isinstance(dsr, dict) else dsr
    pbo = val.get("pbo_cscv", {})
    pbo_v = pbo.get("pbo") if isinstance(pbo, dict) else pbo
    cost = val.get("cost_stress", {})
    return [
        ["symbol", report.get("symbol")],
        ["experiment", report.get("experiment_id")],
        ["champion", (champ.get("genome_hash") or "—")[:16]],
        ["outer-OOS return(s)", json.dumps(oos.get("fold_returns"))],
        ["outer-OOS trades", oos.get("n_trades_total")],
        ["outer-OOS max drawdown",
         f"{100 * oos['max_drawdown']:.2f}%" if oos.get("max_drawdown") is not None else "—"],
        ["deflated Sharpe", dsr_v if dsr_v is not None else "skipped"],
        ["effective trials", val.get("effective_trials", "—")],
        ["PBO (CSCV)", pbo_v if pbo_v is not None else "skipped"],
        ["cost-stress survives 3x spread",
         cost.get("survives_worst_case") if isinstance(cost, dict) else "—"],
        ["lockbox", "SEALED — never opened by a run"],
    ]


def _run_streaming(
    pair_label: str,
    label_to_file: dict[str, str],
    out_root: str,
    seed: int,
    max_bars: int,
    generations: int,
    population: int,
    outer_folds: int,
    outer_test_bars: int,
    inner_folds: int,
    inner_val_bars: int,
    min_train_bars: int,
    lockbox_bars: int,
    commission: float,
    spread_points: float,
    initial_equity: float,
) -> Iterator[tuple[Any, ...]]:
    """Generator: yields (status_md, telemetry_rows, verdict_md, metrics_rows,
    report_json, download_path) tuples as the run progresses."""
    if not pair_label or pair_label not in label_to_file:
        yield ("⚠️ Select a pair first.", [], "", [], {}, None)
        return
    data_file = label_to_file[pair_label]
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = str(Path(out_root) / run_id)

    cfg = RunConfig(
        data_file=data_file,
        out_dir=out_dir,
        seed=int(seed),
        max_bars=int(max_bars) if max_bars else None,
        n_generations=int(generations),
        population_size=int(population),
        n_outer_folds=int(outer_folds),
        outer_test_bars=int(outer_test_bars),
        n_inner_folds=int(inner_folds),
        inner_val_bars=int(inner_val_bars),
        min_train_bars=int(min_train_bars),
        lockbox_bars=int(lockbox_bars),
        commission_pct_notional=float(commission),
        assumed_spread_points=float(spread_points),
        initial_equity=float(initial_equity),
    )

    q: queue.Queue[dict[str, Any]] = queue.Queue()
    result: dict[str, Any] = {}
    error: dict[str, Any] = {}

    def worker() -> None:
        from evoquant.errors import SplitConfigError

        try:
            result["report"] = run_experiment(cfg, on_progress=q.put)
        except SplitConfigError as exc:  # the common "config too big for data" case
            error["message"] = (
                f"{exc}. This pair has fewer bars than the split needs — "
                "increase 'Max bars', or in Advanced settings reduce "
                "'Lockbox bars', 'Outer/Inner folds', test/val bars, or "
                "'Min train bars'."
            )
        except Exception as exc:  # surfaced, never swallowed
            error["message"] = repr(exc)
        finally:
            q.put({"stage": "__done__"})

    threading.Thread(target=worker, daemon=True).start()

    events: list[dict[str, Any]] = []
    while True:
        event = q.get()
        if event.get("stage") == "__done__":
            break
        events.append(event)
        status = f"⏳ **{event['stage']}**"
        if event.get("symbol"):
            status += f" · {event['symbol']}"
        if event.get("generation") is not None:
            status += (f" · fold {event.get('fold')} gen {event['generation']}"
                       f" · HV {float(event.get('hypervolume', 0)):.4f}"
                       f" · feasible {100 * float(event.get('feasibility_rate', 0)):.0f}%")
        yield (status, _telemetry_rows(events), "", [], {}, None)

    if error:
        yield (f"❌ **ERROR** — {error['message']}", _telemetry_rows(events),
               "❌ ERROR", [], {"error": error["message"]}, None)
        return

    report = result["report"]
    verdict = report["verdict"]
    icon = {"SHORTLISTED": "✅", "NO_EDGE_FOUND": "🟡", "REJECTED": "❌"}.get(verdict, "•")
    verdict_md = f"### {icon} {verdict}\n{report.get('verdict_reason', '')}"
    report_path = str(Path(out_dir) / "report.json")
    yield (
        f"✔️ **finished** · verdict **{verdict}**",
        _telemetry_rows(events),
        verdict_md,
        _metrics_rows(report),
        report,
        report_path if Path(report_path).exists() else None,
    )


def _llm_test(provider: str, model: str, api_key: str) -> str:
    from evoquant.llm.client import ChatClient, LLMConfig

    try:
        cfg = LLMConfig(provider=provider, api_key=(api_key or "").strip(),
                        model=model.strip(), timeout_s=20.0)
        rep = ChatClient(cfg).test_connection()
        return (f"✅ OK — {rep['provider']}/{rep['model']} · {rep['latency_s']}s · "
                f"reply: {rep['reply_preview']}")
    except Exception as exc:
        return f"❌ failed: {exc}"


def build_app(data_dir: str, out_root: str) -> gr.Blocks:
    """Construct the Gradio Blocks app (does not launch)."""
    Path(out_root).mkdir(parents=True, exist_ok=True)
    init_rows, init_map = _pairs_table(data_dir)
    init_labels = list(init_map)

    with gr.Blocks(title="evoquant — offline evolution searcher") as app:
        gr.Markdown("# evoquant — universal pair-regime evolution searcher")
        gr.Markdown(DISCLAIMER)
        state_map = gr.State(init_map)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Pairs — drop any ForexSB JSON into the data dir")
                data_dir_box = gr.Textbox(value=data_dir, label="Data directory")
                pairs_df = gr.Dataframe(
                    value=init_rows, headers=["symbol", "file", "bars", "tf"],
                    interactive=False, label="Discovered pairs")
                refresh_btn = gr.Button("🔄 Rescan data dir")
                pair_dd = gr.Dropdown(choices=init_labels, label="Selected pair",
                                      value=init_labels[0] if init_labels else None)

                gr.Markdown("### LLM connection (optional)")
                provider = gr.Dropdown(["openrouter", "openai"], value="openrouter",
                                       label="Provider")
                model = gr.Textbox(label="Model",
                                   placeholder="gpt-4o-mini / ...:free")
                api_key = gr.Textbox(label="API key (not stored)", type="password")
                llm_btn = gr.Button("Test connection")
                llm_msg = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("### Run configuration")
                with gr.Row():
                    seed = gr.Number(value=20260710, label="Seed", precision=0)
                    max_bars = gr.Number(value=12000, label="Max bars (0 = full)", precision=0)
                with gr.Row():
                    generations = gr.Number(value=4, label="Generations", precision=0)
                    population = gr.Number(value=10, label="Population", precision=0)
                with gr.Accordion("Advanced split / cost settings", open=False):
                    with gr.Row():
                        outer_folds = gr.Number(value=3, label="Outer folds", precision=0)
                        outer_test_bars = gr.Number(value=800, label="Outer test bars", precision=0)
                    with gr.Row():
                        inner_folds = gr.Number(value=3, label="Inner folds", precision=0)
                        inner_val_bars = gr.Number(value=400, label="Inner val bars", precision=0)
                    with gr.Row():
                        min_train_bars = gr.Number(
                            value=4000, label="Min train bars", precision=0)
                        lockbox_bars = gr.Number(
                            value=2000, label="Lockbox bars (sealed)", precision=0)
                    with gr.Row():
                        commission = gr.Number(value=0.001, label="Commission (frac notional)")
                        spread_points = gr.Number(value=5.0, label="Assumed spread (points)")
                    initial_equity = gr.Number(value=1_000_000, label="Initial equity")
                run_btn = gr.Button("▶ Run all on selected pair", variant="primary")
                status = gr.Markdown("Idle.")

        gr.Markdown("### Search telemetry")
        telemetry = gr.Dataframe(
            headers=["fold", "gen", "hypervolume", "feasibility", "archive", "intervention"],
            interactive=False, label="per generation")
        with gr.Row():
            verdict = gr.Markdown("")
            metrics = gr.Dataframe(headers=["metric", "value"], interactive=False,
                                   label="Outer-OOS evidence")
        report_json = gr.JSON(label="Full report")
        download = gr.File(label="Download report JSON")

        # ---- wiring ----
        def _refresh(d: str) -> tuple[Any, Any, dict[str, str]]:
            rows, mapping = _pairs_table(d)
            labels = list(mapping)
            return (rows, gr.update(choices=labels, value=labels[0] if labels else None),
                    mapping)

        refresh_btn.click(_refresh, [data_dir_box], [pairs_df, pair_dd, state_map])
        llm_btn.click(_llm_test, [provider, model, api_key], [llm_msg])
        run_btn.click(
            _run_streaming,
            [pair_dd, state_map, gr.State(out_root), seed, max_bars, generations,
             population, outer_folds, outer_test_bars, inner_folds, inner_val_bars,
             min_train_bars, lockbox_bars, commission, spread_points, initial_equity],
            [status, telemetry, verdict, metrics, report_json, download],
        )
    return app


def launch(
    data_dir: str = "data/raw",
    out_root: str = "experiments/gui",
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = True,
    share: bool = False,
) -> None:  # pragma: no cover - launches a blocking server
    app = build_app(data_dir, out_root)
    app.launch(server_name=host, server_port=port, inbrowser=open_browser,
               share=share, quiet=False)
