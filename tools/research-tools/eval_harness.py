"""Formative evaluation harness (standalone).

This is intentionally kept OUTSIDE application repos.

It runs named scenarios against a running EarthLink server by:
- setting tick interval
- resetting + starting the simulation
- subscribing to the world tick WebSocket stream
- collecting tick events for a fixed horizon
- pausing the simulation
- exporting CSV + simple charts (PNG)

No fabricated results: everything comes from live server events.

Usage:
  python test/eval_harness.py --base-url http://127.0.0.1:8000 --ticks 300 --tick-interval 0.25

Outputs:
  test/artifacts/<run_id>/
    ticks.csv
    agent_events.csv
    charts/
      mean_knowledge_score.png
      mean_reward.png
      action_counts.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import websockets


@dataclass
class HarnessConfig:
    base_url: str
    ticks: int
    tick_interval_seconds: float
    connect_timeout_seconds: float
    output_dir: Path


def _now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _ws_url_from_http(base_url: str) -> str:
    base = _normalize_base_url(base_url)
    if base.startswith("https://"):
        return "wss://" + base.removeprefix("https://") + "/ws/world"
    if base.startswith("http://"):
        return "ws://" + base.removeprefix("http://") + "/ws/world"
    # Assume http
    return "ws://" + base + "/ws/world"


async def _collect_ticks(cfg: HarnessConfig) -> list[dict[str, Any]]:
    ws_url = _ws_url_from_http(cfg.base_url)
    ticks: list[dict[str, Any]] = []

    async with websockets.connect(ws_url, open_timeout=cfg.connect_timeout_seconds) as ws:
        while len(ticks) < cfg.ticks:
            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if isinstance(msg, dict) and "tick" in msg:
                ticks.append(msg)

    return ticks


def _http_post(client: httpx.Client, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = client.post(path, json=payload)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _http_get(client: httpx.Client, path: str) -> dict[str, Any]:
    resp = client.get(path)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _flatten_agent_events(tick_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for tick in tick_events:
        tick_id = tick.get("tick")
        time_payload = tick.get("time") or {}
        iso_time = time_payload.get("current_time") or time_payload.get("local_time") or ""
        for ev in tick.get("agent_events", []) or []:
            if not isinstance(ev, dict):
                continue
            flattened.append(
                {
                    "tick": tick_id,
                    "time": iso_time,
                    "agent_id": ev.get("agent_id"),
                    "action": ev.get("action"),
                    "from_location_id": ev.get("from_location_id"),
                    "to_location_id": ev.get("to_location_id"),
                    "reward": ev.get("reward"),
                    "q_value": ev.get("q_value"),
                    "knowledge_score": ev.get("knowledge_score"),
                }
            )
    return flattened


def _series_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return statistics.fmean(values)


def _plot_series(
    *,
    out_path: Path,
    x: list[int],
    y: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_action_counts(out_path: Path, action_counts: dict[str, int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    actions = list(action_counts.keys())
    counts = [action_counts[a] for a in actions]

    plt.figure(figsize=(10, 4))
    plt.bar(actions, counts)
    plt.title("Agent action counts (all agents, all ticks)")
    plt.xlabel("action")
    plt.ylabel("count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _summarize_and_plot(run_dir: Path, tick_events: list[dict[str, Any]]) -> None:
    agent_events = _flatten_agent_events(tick_events)

    # Write raw tick CSV (keep it minimal and stable)
    tick_rows: list[dict[str, Any]] = []
    for t in tick_events:
        time_payload = t.get("time") or {}
        tick_rows.append(
            {
                "tick": t.get("tick"),
                "time_current": time_payload.get("current_time"),
                "time_local": time_payload.get("local_time"),
                "weather_updated": t.get("weather_updated"),
                "wind_updated": t.get("wind_updated"),
                "atmosphere_updated": t.get("atmosphere_updated"),
                "astronomy_updated": t.get("astronomy_updated"),
                "data_feeds_updated": t.get("data_feeds_updated"),
                "agent_event_count": len(t.get("agent_events", []) or []),
                "earth_proxy_resolves": t.get("earth_proxy_resolves"),
            }
        )

    _write_csv(
        run_dir / "ticks.csv",
        tick_rows,
        [
            "tick",
            "time_current",
            "time_local",
            "weather_updated",
            "wind_updated",
            "atmosphere_updated",
            "astronomy_updated",
            "data_feeds_updated",
            "agent_event_count",
            "earth_proxy_resolves",
        ],
    )

    _write_csv(
        run_dir / "agent_events.csv",
        agent_events,
        [
            "tick",
            "time",
            "agent_id",
            "action",
            "from_location_id",
            "to_location_id",
            "reward",
            "q_value",
            "knowledge_score",
        ],
    )

    # Build time series: mean knowledge_score and mean reward per tick
    by_tick: dict[int, list[dict[str, Any]]] = {}
    for ev in agent_events:
        tick = ev.get("tick")
        if isinstance(tick, int):
            by_tick.setdefault(tick, []).append(ev)

    ticks_sorted = sorted(by_tick.keys())
    mean_knowledge: list[float] = []
    mean_reward: list[float] = []

    for tick in ticks_sorted:
        rows = by_tick[tick]
        ks = [v for v in (_to_float(r.get("knowledge_score")) for r in rows) if v is not None]
        rw = [v for v in (_to_float(r.get("reward")) for r in rows) if v is not None]
        mean_knowledge.append(_series_mean(ks))
        mean_reward.append(_series_mean(rw))

    charts_dir = run_dir / "charts"
    _plot_series(
        out_path=charts_dir / "mean_knowledge_score.png",
        x=ticks_sorted,
        y=mean_knowledge,
        title="Mean knowledge score over ticks",
        xlabel="tick",
        ylabel="mean knowledge_score",
    )
    _plot_series(
        out_path=charts_dir / "mean_reward.png",
        x=ticks_sorted,
        y=mean_reward,
        title="Mean reward over ticks",
        xlabel="tick",
        ylabel="mean reward",
    )

    # Action histogram
    action_counts: dict[str, int] = {}
    for ev in agent_events:
        action = ev.get("action")
        if isinstance(action, str) and action:
            action_counts[action] = action_counts.get(action, 0) + 1
    _plot_action_counts(charts_dir / "action_counts.png", action_counts)


async def run(cfg: HarnessConfig) -> Path:
    run_dir = cfg.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "base_url": cfg.base_url,
        "ticks": cfg.ticks,
        "tick_interval_seconds": cfg.tick_interval_seconds,
        "started_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with httpx.Client(base_url=_normalize_base_url(cfg.base_url), timeout=30.0) as client:
        # Configure tick interval
        _http_post(client, "/api/simulation/config", {"tick_interval_seconds": cfg.tick_interval_seconds})

        # Reset to known baseline
        _http_post(client, "/api/simulation/control", {"action": "reset"})

        # Start
        _http_post(client, "/api/simulation/control", {"action": "start"})

        # Collect ticks
        tick_events = await _collect_ticks(cfg)

        # Pause
        _http_post(client, "/api/simulation/control", {"action": "pause"})

        # Save a final snapshot
        try:
            snapshot = _http_get(client, "/api/world/state")
            (run_dir / "world_state.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        except Exception:
            pass

        try:
            agents = client.get("/api/agents?limit=5000").json()
            (run_dir / "agents.json").write_text(json.dumps(agents, indent=2), encoding="utf-8")
        except Exception:
            pass

    _summarize_and_plot(run_dir, tick_events)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--tick-interval", type=float, default=0.25)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--out", default=str(Path("test") / "artifacts" / _now_id()))
    args = parser.parse_args()

    cfg = HarnessConfig(
        base_url=args.base_url,
        ticks=args.ticks,
        tick_interval_seconds=args.tick_interval,
        connect_timeout_seconds=args.connect_timeout,
        output_dir=Path(args.out),
    )

    # Basic output sanity
    os.makedirs(cfg.output_dir, exist_ok=True)

    import asyncio

    run_dir = asyncio.run(run(cfg))
    print(str(run_dir))


if __name__ == "__main__":
    main()
