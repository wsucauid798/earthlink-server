import argparse
import asyncio
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import httpx
import websockets
from websockets.exceptions import ConnectionClosed


@dataclass(frozen=True)
class Scenario:
    name: str
    tick_interval_seconds: float | None
    ticks: int
    pause_at_end: bool = True
    config_change_at_tick: int | None = None
    config_change_to_interval_seconds: float | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_ok(resp: httpx.Response) -> None:
    try:
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from exc


def reset_world(client: httpx.Client, base_url: str) -> None:
    resp = client.post(
        f"{base_url}/api/simulation/control",
        json={"action": "reset"},
    )
    _ensure_ok(resp)


def start_world(client: httpx.Client, base_url: str) -> None:
    resp = client.post(
        f"{base_url}/api/simulation/control",
        json={"action": "start"},
    )
    _ensure_ok(resp)


def pause_world(client: httpx.Client, base_url: str) -> None:
    resp = client.post(
        f"{base_url}/api/simulation/control",
        json={"action": "pause"},
    )
    _ensure_ok(resp)


def set_tick_interval(client: httpx.Client, base_url: str, tick_interval_seconds: float | None) -> None:
    if tick_interval_seconds is None:
        return
    resp = client.post(
        f"{base_url}/api/simulation/config",
        json={"tick_interval_seconds": float(tick_interval_seconds)},
    )
    _ensure_ok(resp)


async def _ws_keepalive(ws: websockets.WebSocketClientProtocol, every_seconds: float = 1.0) -> None:
    while True:
        await asyncio.sleep(every_seconds)
        try:
            await ws.send("ping")
        except Exception:
            return


async def collect_ticks(
    *,
    base_url: str,
    ws_url: str,
    ticks: int,
    out_ndjson_path: Path,
    config_change_at_tick: int | None = None,
    config_change_to_interval_seconds: float | None = None,
    recv_timeout_seconds: float = 30.0,
    progress_every_timeouts: int = 1,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    timeout_count = 0
    scenario_start_tick: int | None = None

    async def _state_snapshot() -> str:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{base_url}/api/world/state")
                resp.raise_for_status()
                data = resp.json()
                tick = data.get("time", {}).get("tick_count")
                resolves = data.get("earth_proxy", {}).get("total_resolves")
                return f"tick={tick} resolves={resolves}"
        except Exception:
            return "tick_state_unavailable"

    last_tick = None
    with out_ndjson_path.open("w", encoding="utf-8", buffering=1) as f:
        while len(collected) < ticks:
            try:
                async with websockets.connect(
                    ws_url,
                    # These runs can have long pauses between server messages.
                    # Disable protocol keepalive timeouts so we don't crash mid-scenario.
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                ) as ws:
                    try:
                        await ws.send("hello")
                    except Exception:
                        pass

                    keepalive = asyncio.create_task(_ws_keepalive(ws, every_seconds=5.0))
                    try:
                        while len(collected) < ticks:
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=recv_timeout_seconds)
                            except asyncio.TimeoutError:
                                timeout_count += 1
                                if progress_every_timeouts > 0 and (timeout_count % progress_every_timeouts == 0):
                                    snap = await _state_snapshot()
                                    print(f"[waiting_for_ticks] {snap}")
                                continue
                            except ConnectionClosed as exc:
                                print(f"[ws_closed] {exc}")
                                break

                            try:
                                tick_data = json.loads(msg)
                            except Exception:
                                continue

                            tick = int(tick_data.get("tick", -1))
                            if last_tick is None:
                                last_tick = tick
                                scenario_start_tick = tick

                            # Protect against reconnect duplicates / out-of-order frames.
                            if last_tick is not None and tick <= last_tick:
                                continue
                            last_tick = tick

                            ticks_since_start = (tick - scenario_start_tick) if scenario_start_tick is not None else 0

                            if (
                                config_change_at_tick is not None
                                and config_change_to_interval_seconds is not None
                                and ticks_since_start >= config_change_at_tick
                            ):
                                with httpx.Client(timeout=30) as client:
                                    set_tick_interval(client, base_url, float(config_change_to_interval_seconds))
                                config_change_at_tick = None

                            collected.append(tick_data)
                            f.write(json.dumps(tick_data, ensure_ascii=False))
                            f.write("\n")
                            f.flush()

                            if len(collected) == 1 or len(collected) % 5 == 0:
                                print(
                                    f"[tick] tick={tick} (+{ticks_since_start}) "
                                    f"collected={len(collected)}/{ticks}"
                                )
                    finally:
                        keepalive.cancel()
                        try:
                            await keepalive
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
            except Exception as exc:
                print(f"[ws_connect_error] {exc}")
                await asyncio.sleep(1.0)

    return collected


def summarize_ticks(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in ticks:
        agent_events = t.get("agent_events") or []
        rewards = [e.get("reward") for e in agent_events if isinstance(e, dict) and isinstance(e.get("reward"), (int, float))]
        knowledge_scores = [
            e.get("knowledge_score")
            for e in agent_events
            if isinstance(e, dict) and isinstance(e.get("knowledge_score"), (int, float))
        ]
        rows.append(
            {
                "tick": int(t.get("tick", -1)),
                "agent_events": int(len(agent_events)),
                "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
                "mean_knowledge_score": (sum(knowledge_scores) / len(knowledge_scores)) if knowledge_scores else 0.0,
                "earth_proxy_resolves": int(t.get("earth_proxy_resolves", 0) or 0),
                "weather_updated": bool(t.get("weather_updated", False)),
                "wind_updated": bool(t.get("wind_updated", False)),
                "atmosphere_updated": bool(t.get("atmosphere_updated", False)),
                "astronomy_updated": bool(t.get("astronomy_updated", False)),
                "data_feeds_updated": bool(t.get("data_feeds_updated", False)),
            }
        )
    rows.sort(key=lambda r: r["tick"])
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_series(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return

    xs = [r["tick"] for r in rows]

    save_eps = os.environ.get("EARTHLINK_SAVE_EPS") == "1"

    def _save(path: Path) -> None:
        if save_eps and path.suffix.lower() == ".png":
            plt.savefig(path.with_suffix(".eps"), format="eps")
        plt.savefig(path, dpi=160)
        plt.close()

    def _plot(y_key: str, title: str, filename: str) -> None:
        ys = [r[y_key] for r in rows]
        plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, linewidth=1.5)
        plt.title(title)
        plt.xlabel("tick")
        plt.ylabel(y_key)
        plt.tight_layout()
        _save(out_dir / filename)

    _plot("agent_events", "Agent events per tick", "agent_events.png")
    _plot("mean_reward", "Mean reward per tick", "mean_reward.png")
    _plot("mean_knowledge_score", "Mean knowledge score per tick", "mean_knowledge_score.png")

    # Cumulative resolves
    resolves = [r["earth_proxy_resolves"] for r in rows]
    cum = []
    total = 0
    for v in resolves:
        total += int(v)
        cum.append(total)
    plt.figure(figsize=(10, 4))
    plt.plot(xs, cum, linewidth=1.5)
    plt.title("Cumulative earth_proxy_resolves")
    plt.xlabel("tick")
    plt.ylabel("cumulative_resolves")
    plt.tight_layout()
    _save(out_dir / "earth_proxy_resolves_cum.png")


async def run_scenario(
    base_url: str,
    ws_url: str,
    scenario: Scenario,
    out_root: Path,
    *,
    reset_before: bool,
    recv_timeout_seconds: float,
) -> None:
    out_dir = out_root / scenario.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Control: (optional reset) -> configure -> start
    with httpx.Client(timeout=30) as client:
        if reset_before:
            reset_world(client, base_url)
        set_tick_interval(client, base_url, scenario.tick_interval_seconds)
        start_world(client, base_url)

    ndjson_path = out_dir / "ticks.ndjson"
    ticks = await collect_ticks(
        base_url=base_url,
        ws_url=ws_url,
        ticks=scenario.ticks,
        out_ndjson_path=ndjson_path,
        config_change_at_tick=scenario.config_change_at_tick,
        config_change_to_interval_seconds=scenario.config_change_to_interval_seconds,
        recv_timeout_seconds=recv_timeout_seconds,
    )

    if scenario.pause_at_end:
        with httpx.Client(timeout=30) as client:
            pause_world(client, base_url)

    rows = summarize_ticks(ticks)
    write_csv(rows, out_dir / "summary.csv")
    plot_series(rows, out_dir)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("EARTHLINK_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--ws-url", default=os.environ.get("EARTHLINK_WS_URL", "ws://localhost:8000/ws/world"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "artifacts" / _utc_stamp()))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a shorter suite (faster) while still producing CSV/PNG artifacts.",
    )
    parser.add_argument(
        "--reset-between",
        action="store_true",
        help="Reset the world before each scenario (slower; may re-trigger heavy first-tick work).",
    )
    parser.add_argument(
        "--recv-timeout-seconds",
        type=float,
        default=float(os.environ.get("EARTHLINK_RECV_TIMEOUT_SECONDS", "30")),
        help="WebSocket recv timeout while waiting for ticks (prints progress and retries).",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.quick:
        suite = [
            Scenario(name="baseline_12", tick_interval_seconds=1.0, ticks=12),
            Scenario(name="fast_12", tick_interval_seconds=0.2, ticks=12),
            Scenario(name="slow_10", tick_interval_seconds=2.0, ticks=10),
            Scenario(
                name="step_change_18",
                tick_interval_seconds=1.0,
                ticks=18,
                config_change_at_tick=5,
                config_change_to_interval_seconds=0.2,
            ),
            Scenario(
                name="step_change_back_24",
                tick_interval_seconds=0.2,
                ticks=24,
                config_change_at_tick=8,
                config_change_to_interval_seconds=1.0,
            ),
        ]
    else:
        suite = [
            Scenario(name="baseline_200", tick_interval_seconds=1.0, ticks=200),
            Scenario(name="fast_200", tick_interval_seconds=0.2, ticks=200),
            Scenario(name="slow_120", tick_interval_seconds=2.0, ticks=120),
            Scenario(name="step_change_240", tick_interval_seconds=1.0, ticks=240, config_change_at_tick=60, config_change_to_interval_seconds=0.2),
            Scenario(name="step_change_back_300", tick_interval_seconds=0.2, ticks=300, config_change_at_tick=120, config_change_to_interval_seconds=1.0),
        ]

    # Quick reachability check
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{args.base_url}/api/version")
            _ensure_ok(resp)
    except Exception as exc:
        raise RuntimeError(
            f"Server not reachable at {args.base_url}. "
            f"Start earthlink-server, then rerun. Original error: {exc}"
        )

    for scenario in suite:
        started = time.time()
        print(f"==> {scenario.name}")
        await run_scenario(
            args.base_url,
            args.ws_url,
            scenario,
            out_root,
            reset_before=args.reset_between,
            recv_timeout_seconds=args.recv_timeout_seconds,
        )
        elapsed = time.time() - started
        print(f"<== {scenario.name} done in {elapsed:.1f}s")

    print(f"Artifacts: {out_root}")


if __name__ == "__main__":
    asyncio.run(main())
