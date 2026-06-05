"""EarthLink SVW -- Empirical Evaluation Suite

Runs a series of evaluation scenarios against a live EarthLink server,
collecting rich agent-level and system-level data via REST + WebSocket,
then produces publication-quality charts for the research paper.

Usage:
    python eval_suite.py --quick          # fast smoke-test (~2 min)
    python eval_suite.py                  # full evaluation (~10 min)
    python eval_suite.py --base-url http://host:port
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
# websockets kept as optional import in case WS mode is re-enabled later

# ---------------------------------------------------------------------------
# Chart imports (deferred to avoid import-time crash if not installed)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="white", font_scale=1.1, palette="muted")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Publication style defaults
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    # Type 42 (TrueType) fonts in both PS/EPS and PDF outputs.
    # EPS: selectable text, no bitmapped Type 3 fonts (required by Springer).
    # PDF: renders correctly in all viewers (Acrobat, browsers, LaTeX pdflatex).
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})


# When enabled, charts are saved as both .png and .eps (same stem).
# This does NOT modify or delete any existing PNGs; it only adds EPS files.
SAVE_EPS = False

# When enabled, charts are also saved as PDF (same stem, .pdf suffix).
# PDF with pdf.fonttype=42 renders correctly in all viewers including Windows.
# Springer pdflatex accepts PDF figures via \includegraphics.
SAVE_PDF = False

# Control what we write during a run. These allow re-running against an existing
# artifacts directory to backfill missing EPS without touching existing PNG/data.
SAVE_PNG = True
EPS_ONLY_MISSING = False
WRITE_DATA = True
WRITE_REPORT = True
PERSIST_SNAPSHOTS = True

# Snapshot polling can be slow under load; keep this configurable.
SNAPSHOT_HTTP_TIMEOUT_SECONDS = 60.0


# ============================= Data Structures =============================

@dataclass
class Scenario:
    name: str
    tick_interval_seconds: float | None
    ticks: int
    pause_at_end: bool = True
    config_change_at_tick: int | None = None
    config_change_to_interval_seconds: float | None = None


@dataclass
class Snapshot:
    wall_time: float
    tick: int
    agents: list[dict[str, Any]]
    earth_proxy: dict[str, Any] | None
    config: dict[str, Any] | None


@dataclass
class TickRecord:
    tick: int
    wall_time: str
    agent_events: list[dict[str, Any]]
    earth_proxy_resolves: int
    agent_phase_exceeded: bool
    weather_updated: bool
    wind_updated: bool
    atmosphere_updated: bool
    astronomy_updated: bool
    data_feeds_updated: bool


@dataclass
class ScenarioResult:
    name: str
    ticks: list[TickRecord]
    snapshots: list[Snapshot]
    start_wall: float
    end_wall: float


# ============================= Server Helpers ==============================

def _ensure_ok(resp: httpx.Response) -> None:
    try:
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from exc


def reset_world(client: httpx.Client, base: str) -> None:
    _ensure_ok(client.post(f"{base}/api/simulation/control", json={"action": "reset"}))


def start_world(client: httpx.Client, base: str) -> None:
    _ensure_ok(client.post(f"{base}/api/simulation/control", json={"action": "start"}))


def pause_world(client: httpx.Client, base: str) -> None:
    _ensure_ok(client.post(f"{base}/api/simulation/control", json={"action": "pause"}))


def set_tick_interval(client: httpx.Client, base: str, interval: float | None) -> None:
    if interval is None:
        return
    _ensure_ok(client.post(f"{base}/api/simulation/config", json={"tick_interval_seconds": float(interval)}))


def fetch_snapshot(client: httpx.Client, base: str) -> Snapshot:
    resp = client.get(f"{base}/api/eval/snapshot", timeout=SNAPSHOT_HTTP_TIMEOUT_SECONDS)
    _ensure_ok(resp)
    data = resp.json()
    return Snapshot(
        wall_time=time.time(),
        tick=data["tick"],
        agents=data["agents"],
        earth_proxy=data.get("earth_proxy"),
        config=data.get("config"),
    )


# ============================= Collection ==================================

async def collect_scenario(
    base_url: str,
    ws_url: str,
    scenario: Scenario,
    *,
    reset_before: bool,
    snapshot_every_ticks: int = 5,
    recv_timeout: float = 30.0,
    poll_interval_seconds: float = 3.0,
) -> ScenarioResult:
    """Run one scenario via REST polling.

    Since adapter calls block the event loop (synchronous httpx inside asyncio),
    WS broadcasts are unreliable. We use periodic REST polling of /api/eval/snapshot
    instead, detecting new ticks by comparing tick counts.
    """

    # Setup
    with httpx.Client(timeout=30) as client:
        if reset_before:
            reset_world(client, base_url)
        set_tick_interval(client, base_url, scenario.tick_interval_seconds)
        start_world(client, base_url)

    start_wall = time.time()
    ticks: list[TickRecord] = []
    snapshots: list[Snapshot] = []
    scenario_start_tick: int | None = None
    last_tick: int | None = None
    last_snapshot_tick: int = -1

    while len(ticks) < scenario.ticks:
        await asyncio.sleep(poll_interval_seconds)
        try:
            with httpx.Client(timeout=30) as c:
                snap = fetch_snapshot(c, base_url)
        except Exception as exc:
            print(f"  [poll_error] {exc}", flush=True)
            continue

        tick_num = snap.tick
        if last_tick is not None and tick_num <= last_tick:
            continue  # no new tick yet

        if scenario_start_tick is None:
            scenario_start_tick = tick_num

        last_tick = tick_num
        ticks_since = tick_num - scenario_start_tick

        # Mid-scenario config change
        if (
            scenario.config_change_at_tick is not None
            and scenario.config_change_to_interval_seconds is not None
            and ticks_since >= scenario.config_change_at_tick
        ):
            try:
                with httpx.Client(timeout=30) as c:
                    set_tick_interval(c, base_url, scenario.config_change_to_interval_seconds)
                print(f"  [config_change] tick_interval -> {scenario.config_change_to_interval_seconds}s at tick {tick_num}", flush=True)
            except Exception as exc:
                print(f"  [config_change_error] {exc}", flush=True)
            scenario.config_change_at_tick = None  # fire once

        # Build synthetic TickRecord from snapshot (earth_proxy has total_resolves)
        total_resolves = (snap.earth_proxy or {}).get("total_resolves", 0)
        tr = TickRecord(
            tick=tick_num,
            wall_time=datetime.now(timezone.utc).isoformat(),
            agent_events=[],  # not available via REST polling
            earth_proxy_resolves=int(total_resolves or 0),
            agent_phase_exceeded=False,  # unknown via REST
            weather_updated=False,
            wind_updated=False,
            atmosphere_updated=False,
            astronomy_updated=False,
            data_feeds_updated=False,
        )
        ticks.append(tr)
        snapshots.append(snap)

        n = len(ticks)
        if n == 1 or n % 5 == 0:
            ks = [a.get("knowledge_score", 0) for a in snap.agents]
            vl = [a.get("visited_locations", 0) for a in snap.agents]
            mean_ks = sum(ks) / len(ks) if ks else 0
            mean_vl = sum(vl) / len(vl) if vl else 0
            print(
                f"  [tick] {tick_num} (+{ticks_since}) collected={n}/{scenario.ticks}"
                f"  knowledge={mean_ks:.1f}  visited={mean_vl:.1f}",
                flush=True,
            )

    # Final snapshot
    try:
        with httpx.Client(timeout=30) as c:
            snapshots.append(fetch_snapshot(c, base_url))
    except Exception:
        pass

    if scenario.pause_at_end:
        try:
            with httpx.Client(timeout=30) as c:
                pause_world(c, base_url)
        except Exception:
            pass

    return ScenarioResult(
        name=scenario.name,
        ticks=ticks,
        snapshots=snapshots,
        start_wall=start_wall,
        end_wall=time.time(),
    )


# ============================= Analysis ====================================

def _agent_time_series(snapshots: list[Snapshot]) -> dict[str, list[dict]]:
    """Build per-agent time series from snapshots."""
    series: dict[str, list[dict]] = {}
    for snap in snapshots:
        for a in snap.agents:
            aid = a["id"]
            if aid not in series:
                series[aid] = []
            series[aid].append({
                "tick": snap.tick,
                "wall_time": snap.wall_time,
                "knowledge_score": a.get("knowledge_score", 0),
                "visited_locations": a.get("visited_locations", 0),
                "energy": a.get("energy", 100),
                "last_action": a.get("last_action", ""),
                "last_reward": a.get("last_reward", 0),
                "location_id": a.get("location_id", 0),
                "goal_kind": (a.get("goal") or {}).get("kind", "none"),
            })
    return series


def _aggregate_snapshots(snapshots: list[Snapshot]) -> list[dict]:
    """Aggregate all agents per snapshot into summary rows."""
    rows = []
    for snap in snapshots:
        agents = snap.agents
        if not agents:
            continue
        ks = [a.get("knowledge_score", 0) for a in agents]
        vl = [a.get("visited_locations", 0) for a in agents]
        rw = [a.get("last_reward", 0) for a in agents]
        en = [a.get("energy", 100) for a in agents]
        actions = {}
        goals = {}
        locations = set()
        for a in agents:
            act = a.get("last_action", "unknown")
            actions[act] = actions.get(act, 0) + 1
            gk = (a.get("goal") or {}).get("kind", "none")
            goals[gk] = goals.get(gk, 0) + 1
            locations.add(a.get("location_id", 0))
        rows.append({
            "tick": snap.tick,
            "wall_time": snap.wall_time,
            "n_agents": len(agents),
            "mean_knowledge": np.mean(ks),
            "std_knowledge": np.std(ks),
            "median_knowledge": np.median(ks),
            "max_knowledge": max(ks),
            "mean_visited": np.mean(vl),
            "std_visited": np.std(vl),
            "max_visited": max(vl),
            "mean_reward": np.mean(rw),
            "std_reward": np.std(rw),
            "mean_energy": np.mean(en),
            "unique_locations_occupied": len(locations),
            "actions": actions,
            "goals": goals,
            "earth_proxy_resolves": (snap.earth_proxy or {}).get("total_resolves", 0),
        })
    return rows


# ============================= Charting ====================================

COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#059669",
    "warn": "#D97706",
    "danger": "#DC2626",
    "grey": "#6B7280",
    "light": "#93C5FD",
}

ACTION_COLORS = {
    "move": "#2563EB",
    "observe": "#059669",
    "rest": "#D97706",
    "spawned": "#6B7280",
}

GOAL_COLORS = {
    "explore": "#2563EB",
    "investigate_gap": "#7C3AED",
    "recover": "#D97706",
    "seek_agent": "#059669",
    "none": "#D1D5DB",
}


def _save(fig, path: Path) -> None:
    wrote_any = False

    # EPS (optionally only if missing)
    if SAVE_EPS and path.suffix.lower() == ".png":
        eps_path = path.with_suffix(".eps")
        if (not EPS_ONLY_MISSING) or (not eps_path.exists()):
            fig.savefig(eps_path, format="eps")
            wrote_any = True
            print(f"    -> {eps_path.name}")

    # PDF — renders correctly in all viewers; accepted by Springer pdflatex
    if SAVE_PDF and path.suffix.lower() == ".png":
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, format="pdf")
        wrote_any = True
        print(f"    -> {pdf_path.name}")

    # PNG (optionally disabled to avoid touching existing PNGs)
    if SAVE_PNG:
        fig.savefig(path)
        wrote_any = True
        print(f"    -> {path.name}")

    plt.close(fig)


def chart_tick_cadence(result: ScenarioResult, out: Path) -> None:
    """Wall-clock interval between consecutive ticks."""
    if len(result.ticks) < 3:
        return
    intervals = []
    for i in range(1, len(result.ticks)):
        # Parse wall time
        try:
            t1 = datetime.fromisoformat(result.ticks[i - 1].wall_time)
            t2 = datetime.fromisoformat(result.ticks[i].wall_time)
            intervals.append((t2 - t1).total_seconds())
        except Exception:
            pass
    if not intervals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})

    # Time series
    ax = axes[0]
    xs = list(range(len(intervals)))
    med = np.median(intervals)
    ax.plot(xs, intervals, linewidth=1.2, color=COLORS["primary"], alpha=0.8)
    ax.axhline(y=med, color=COLORS["accent"], linestyle="--", linewidth=1, label=f"Median = {med:.2f}s")
    _apply_axis_style(ax, xs,
                      xlabel="Interval index (consecutive ticks)",
                      ylabel="Wall-clock interval (s)",
                      title="Tick Cadence Over Time")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Histogram
    ax2 = axes[1]
    ax2.hist(intervals, bins=min(25, len(intervals)), color=COLORS["primary"], alpha=0.8, edgecolor="white")
    ax2.axvline(med, color=COLORS["accent"], linestyle="--", linewidth=1.5, label=f"Median")
    ax2.set_xlabel("Interval (s)", fontsize=10, labelpad=6)
    ax2.set_ylabel("Count", fontsize=10, labelpad=6)
    ax2.set_title("Distribution", fontsize=11, fontweight="bold", pad=8)
    _springer_bar_ax(ax2)

    fig.suptitle(f"Tick Cadence — {result.name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "tick_cadence.png")


def _seq_x(data: list) -> tuple[list[int], list[str]]:
    """Return sequential 0-based x positions and tick-number string labels."""
    xs = list(range(len(data)))
    return xs, xs  # caller can use tick numbers as labels if preferred


def _apply_axis_style(ax, xs: list[int], xlabel: str, ylabel: str, title: str) -> None:
    """Apply Springer-style axis formatting: clean white background, no grid, ticks only on bottom/left."""
    n = len(xs)
    ax.set_xlim(-0.3, n - 0.7)
    # Show at most 8 evenly-spaced tick marks on x-axis
    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    if (n - 1) not in ticks:
        ticks.append(n - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t + 1) for t in ticks])  # 1-based snapshot labels
    ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=9, direction="out", length=4)
    # No background grid — Springer style
    ax.grid(False)
    ax.set_facecolor("white")
    # Only bottom and left spines visible
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)


def chart_knowledge_growth(agg: list[dict], out: Path, title_suffix: str = "") -> None:
    """Mean knowledge score with +/- 1 std band over snapshots."""
    if len(agg) < 2:
        return
    xs = list(range(len(agg)))
    mean = [r["mean_knowledge"] for r in agg]
    std  = [r["std_knowledge"]  for r in agg]
    upper = [m + s for m, s in zip(mean, std)]
    lower = [max(0.0, m - s) for m, s in zip(mean, std)]
    maxk  = [r["max_knowledge"] for r in agg]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(xs, lower, upper, alpha=0.25, color=COLORS["primary"], label="\u00b11 SD")
    ax.plot(xs, mean, linewidth=2.0, color=COLORS["primary"], label="Mean (45 agents)")
    ax.plot(xs, maxk, linewidth=1.2, color=COLORS["secondary"], linestyle="--", label="Max agent")
    ax.set_ylim(bottom=0)
    _apply_axis_style(ax, xs,
                      xlabel="Snapshot index",
                      ylabel="Knowledge score",
                      title=f"Agent Knowledge Growth{title_suffix}")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out / "knowledge_growth.png")


def chart_exploration_frontier(agg: list[dict], agent_series: dict, out: Path, title_suffix: str = "") -> None:
    """Unique locations visited over snapshots (aggregate + individual traces)."""
    if len(agg) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: aggregate
    ax = axes[0]
    xs = list(range(len(agg)))
    mean_vis = [r["mean_visited"] for r in agg]
    std_vis  = [r["std_visited"]  for r in agg]
    upper    = [m + s for m, s in zip(mean_vis, std_vis)]
    lower    = [max(0.0, m - s) for m, s in zip(mean_vis, std_vis)]
    max_vis  = [r["max_visited"] for r in agg]

    ax.fill_between(xs, lower, upper, alpha=0.25, color=COLORS["primary"], label="\u00b11 SD")
    ax.plot(xs, mean_vis, linewidth=2.0, color=COLORS["primary"], label="Mean (45 agents)")
    ax.plot(xs, max_vis,  linewidth=1.2, color=COLORS["secondary"], linestyle="--", label="Max agent")
    ax.set_ylim(bottom=0)
    _apply_axis_style(ax, xs,
                      xlabel="Snapshot index",
                      ylabel="Unique Locations Visited",
                      title="Aggregate Exploration Coverage")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Right: individual agent traces (sample up to 10)
    ax2 = axes[1]
    sample_ids = sorted(agent_series.keys())[:10]
    n_pts = len(agg)
    for aid in sample_ids:
        pts = agent_series[aid]
        pt_xs = list(range(len(pts)))
        ax2.plot(pt_xs, [p["visited_locations"] for p in pts],
                 linewidth=0.9, alpha=0.75, label=aid)
    ax2.set_ylim(bottom=0)
    _apply_axis_style(ax2, list(range(n_pts)),
                      xlabel="Snapshot index",
                      ylabel="Unique Locations Visited",
                      title="Per-Agent Traces (sample of 10)")
    if sample_ids:
        ax2.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)

    fig.suptitle(f"Exploration Frontier{title_suffix}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "exploration_frontier.png")


def chart_reward_trajectory(agg: list[dict], out: Path, title_suffix: str = "") -> None:
    """Mean reward with +/-1 SD band over snapshots."""
    if len(agg) < 2:
        return
    xs     = list(range(len(agg)))
    mean_r = [r["mean_reward"] for r in agg]
    std_r  = [r["std_reward"]  for r in agg]
    upper  = [m + s for m, s in zip(mean_r, std_r)]
    lower  = [m - s for m, s in zip(mean_r, std_r)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(xs, lower, upper, alpha=0.25, color=COLORS["accent"], label="\u00b11 SD")
    ax.plot(xs, mean_r, linewidth=2.0, color=COLORS["accent"], label="Mean reward")
    ax.axhline(0, color=COLORS["grey"], linewidth=0.8, linestyle=":", label="Zero baseline")
    _apply_axis_style(ax, xs,
                      xlabel="Snapshot (collected in order)",
                      ylabel="Mean Reward (per-agent, per-tick)",
                      title=f"Reward Trajectory{title_suffix}")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out / "reward_trajectory.png")


def chart_action_distribution(agg: list[dict], out: Path, title_suffix: str = "") -> None:
    """Stacked area chart of action types over time."""
    if len(agg) < 2:
        return

    all_actions = set()
    for r in agg:
        all_actions.update(r["actions"].keys())
    all_actions = sorted(all_actions)

    xs = list(range(len(agg)))
    stacks = {a: [] for a in all_actions}
    for r in agg:
        total = sum(r["actions"].values())
        for a in all_actions:
            count = r["actions"].get(a, 0)
            stacks[a].append(count / total * 100 if total > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Stacked area
    ax = axes[0]
    colors = [ACTION_COLORS.get(a, COLORS["grey"]) for a in all_actions]
    ax.stackplot(xs, *[stacks[a] for a in all_actions], labels=all_actions, colors=colors, alpha=0.8)
    _apply_axis_style(ax, xs,
                      xlabel="Snapshot (collected in order)",
                      ylabel="Share of agents (%)",
                      title="Action Distribution Over Time")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 100)

    # Final snapshot bar chart
    ax2 = axes[1]
    if agg:
        final = agg[-1]["actions"]
        acts = sorted(final.keys())
        vals = [final[a] for a in acts]
        cols = [ACTION_COLORS.get(a, COLORS["grey"]) for a in acts]
        bars = ax2.bar(acts, vals, color=cols, edgecolor="white", linewidth=0.8)
        ax2.set_xlabel("Action", fontsize=10, labelpad=6)
        ax2.set_ylabel("Agent Count", fontsize=10, labelpad=6)
        ax2.set_title("Final Action Breakdown", fontsize=11, fontweight="bold", pad=8)
        ax2.tick_params(axis="both", labelsize=9, direction="out", length=4)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(val), ha="center", va="bottom", fontsize=9)
        _springer_bar_ax(ax2)

    fig.suptitle(f"Agent Actions{title_suffix}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "action_distribution.png")


def chart_goal_distribution(agg: list[dict], out: Path, title_suffix: str = "") -> None:
    """Stacked area chart of goal types over time."""
    if len(agg) < 2:
        return
    all_goals = set()
    for r in agg:
        all_goals.update(r["goals"].keys())
    all_goals = sorted(all_goals)

    xs = list(range(len(agg)))
    stacks = {g: [] for g in all_goals}
    for r in agg:
        total = sum(r["goals"].values())
        for g in all_goals:
            count = r["goals"].get(g, 0)
            stacks[g].append(count / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [GOAL_COLORS.get(g, COLORS["grey"]) for g in all_goals]
    ax.stackplot(xs, *[stacks[g] for g in all_goals], labels=all_goals, colors=colors, alpha=0.8)
    _apply_axis_style(ax, xs,
                      xlabel="Snapshot (collected in order)",
                      ylabel="Share of agents (%)",
                      title=f"Goal Distribution Over Time{title_suffix}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, out / "goal_distribution.png")


def chart_energy_distribution(snapshots: list[Snapshot], out: Path, title_suffix: str = "") -> None:
    """Violin/box plot of agent energy at first and last snapshot."""
    if len(snapshots) < 2:
        return
    first = snapshots[0]
    last = snapshots[-1]

    e_first = [a.get("energy", 100) for a in first.agents]
    e_last = [a.get("energy", 100) for a in last.agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot([e_first, e_last], positions=[1, 2], showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(COLORS["primary"])
        pc.set_alpha(0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Snapshot 1\n(start)", f"Snapshot {len(snapshots)}\n(end)"], fontsize=9)
    ax.set_ylabel("Energy", fontsize=10, labelpad=6)
    ax.set_xlabel("Scenario Position", fontsize=10, labelpad=6)
    ax.set_title(f"Agent Energy Distribution{title_suffix}", fontsize=11, fontweight="bold", pad=8)
    _springer_bar_ax(ax)
    fig.tight_layout()
    _save(fig, out / "energy_distribution.png")


def chart_earth_proxy_throughput(agg: list[dict], out: Path, title_suffix: str = "") -> None:
    """Cumulative Earth proxy resolves over snapshots."""
    if len(agg) < 2:
        return
    xs       = list(range(len(agg)))
    resolves = [r["earth_proxy_resolves"] for r in agg]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, resolves, linewidth=2.0, color=COLORS["secondary"], label="Cumulative resolves")
    ax.fill_between(xs, 0, resolves, alpha=0.15, color=COLORS["secondary"])
    ax.set_ylim(bottom=0)
    _apply_axis_style(ax, xs,
                      xlabel="Snapshot (collected in order)",
                      ylabel="Total Earth-proxy resolves (cumulative)",
                      title=f"Earth Proxy Evidence Throughput{title_suffix}")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out / "earth_proxy_throughput.png")


def chart_geographic_spread(snapshots: list[Snapshot], out: Path, title_suffix: str = "") -> None:
    """Number of distinct locations occupied by at least one agent, per snapshot."""
    if len(snapshots) < 2:
        return
    xs     = list(range(len(snapshots)))
    unique = [len(set(a.get("location_id", 0) for a in s.agents)) for s in snapshots]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, unique, linewidth=2.0, color=COLORS["warn"],
            marker="o", markersize=4, label="Occupied locations")
    ax.set_ylim(bottom=0)
    _apply_axis_style(ax, xs,
                      xlabel="Snapshot (collected in order)",
                      ylabel="Distinct occupied locations (count)",
                      title=f"Geographic Spread of Agents{title_suffix}")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out / "geographic_spread.png")


def chart_adapter_performance(snapshots: list[Snapshot], out: Path, title_suffix: str = "") -> None:
    """Per-adapter latency and call counts from the last snapshot's earth_proxy stats."""
    if not snapshots:
        return
    ep = snapshots[-1].earth_proxy
    if not ep:
        return

    # Per-adapter stats live under "adapters" key
    adapter_stats = ep.get("adapters", {})
    if not adapter_stats:
        # Fallback: scan top-level for dicts with "calls"
        adapter_stats = {k: v for k, v in ep.items() if isinstance(v, dict) and "calls" in v}
    if not adapter_stats:
        return

    # Sort by call count descending, take top 15
    sorted_adapters = sorted(adapter_stats.items(), key=lambda x: x[1].get("calls", 0), reverse=True)[:15]
    if not sorted_adapters:
        return

    names = [a[0] for a in sorted_adapters]
    calls = [a[1].get("calls", 0) for a in sorted_adapters]
    latency = [a[1].get("avg_latency_ms", 0) for a in sorted_adapters]
    timeouts = [a[1].get("timeouts", 0) for a in sorted_adapters]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    y_pos = list(range(len(names)))

    # Calls
    ax = axes[0]
    ax.barh(y_pos, calls, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Total Calls", fontsize=10, labelpad=6)
    ax.set_title("Adapter Call Volume", fontsize=11, fontweight="bold", pad=8)
    ax.invert_yaxis()
    _springer_bar_ax(ax)

    # Latency
    ax2 = axes[1]
    colors_lat = [COLORS["danger"] if l > 5000 else COLORS["warn"] if l > 1000 else COLORS["accent"] for l in latency]
    ax2.barh(y_pos, latency, color=colors_lat, alpha=0.85, edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Avg Latency (ms)", fontsize=10, labelpad=6)
    ax2.set_title("Adapter Latency", fontsize=11, fontweight="bold", pad=8)
    ax2.invert_yaxis()
    _springer_bar_ax(ax2)

    # Timeouts
    ax3 = axes[2]
    ax3.barh(y_pos, timeouts, color=COLORS["danger"], alpha=0.85, edgecolor="white")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel("Timeout Count", fontsize=10, labelpad=6)
    ax3.set_title("Adapter Timeouts", fontsize=11, fontweight="bold", pad=8)
    ax3.invert_yaxis()
    _springer_bar_ax(ax3)

    fig.suptitle(f"Earth Adapter Performance{title_suffix}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "adapter_performance.png")


def _springer_bar_ax(ax) -> None:
    """Apply Springer style to a bar/scatter axes (no grid, clean spines)."""
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, direction="out", length=4)


def chart_system_health(result: ScenarioResult, out: Path) -> None:
    """Agent phase budget and data-refresh events over collected ticks."""
    if len(result.ticks) < 3:
        return

    xs       = list(range(len(result.ticks)))
    exceeded = [1 if t.agent_phase_exceeded else 0 for t in result.ticks]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.fill_between(xs, 0, exceeded, step="mid", alpha=0.45, color=COLORS["danger"],
                    label="Phase exceeded")
    ax.set_ylabel("Agent phase\nbudget", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["OK", "Exceeded"], fontsize=9)
    ax.set_ylim(-0.1, 1.4)
    ax.set_title("Agent Phase Budget Exceedance", fontsize=11, fontweight="bold", pad=6)
    _springer_bar_ax(ax)

    ax2 = axes[1]
    domains       = ["weather", "wind", "atmosphere", "astronomy", "data_feeds"]
    domain_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                     COLORS["warn"], COLORS["danger"]]
    for i, domain in enumerate(domains):
        updates = [1 if getattr(t, f"{domain}_updated", False) else 0 for t in result.ticks]
        ax2.scatter(
            [x for x, u in zip(xs, updates) if u],
            [i + 1] * sum(updates),
            s=14, color=domain_colors[i], label=domain, alpha=0.85,
        )
    step = max(1, len(xs) // 8)
    ticks = list(range(0, len(xs), step))
    if (len(xs) - 1) not in ticks:
        ticks.append(len(xs) - 1)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([str(t + 1) for t in ticks], fontsize=9)
    ax2.set_xlabel("Snapshot (collected in order)", fontsize=10, labelpad=6)
    ax2.set_ylabel("Domain", fontsize=10, labelpad=6)
    ax2.set_yticks(range(1, len(domains) + 1))
    ax2.set_yticklabels(domains, fontsize=9)
    ax2.set_title("Data Refresh Events by Domain", fontsize=11, fontweight="bold", pad=6)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.set_xlim(-0.5, len(xs) - 0.5)
    _springer_bar_ax(ax2)

    fig.suptitle(f"System Health — {result.name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "system_health.png")


# ============================= Cross-Scenario Comparison ====================

def chart_cross_scenario_comparison(results: list[ScenarioResult], out: Path) -> None:
    """Compare key metrics across all scenarios in one multi-panel figure."""
    if len(results) < 2:
        return

    names = [r.name for r in results]
    n = len(names)

    # Compute final-snapshot metrics for each scenario
    final_knowledge = []
    final_visited = []
    final_reward = []
    final_resolves = []
    tick_counts = []

    for r in results:
        agg = _aggregate_snapshots(r.snapshots)
        if agg:
            final_knowledge.append(agg[-1]["mean_knowledge"])
            final_visited.append(agg[-1]["mean_visited"])
            final_reward.append(agg[-1]["mean_reward"])
            final_resolves.append(agg[-1]["earth_proxy_resolves"])
        else:
            final_knowledge.append(0)
            final_visited.append(0)
            final_reward.append(0)
            final_resolves.append(0)
        tick_counts.append(len(r.ticks))

    def _bar_panel(ax, vals, color, ylabel, title, fmt="{:.1f}"):
        bars = ax.bar(range(n), vals, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9, labelpad=5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylim(bottom=0, top=max(vals) * 1.18 if vals else 1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    fmt.format(val), ha="center", va="bottom", fontsize=8)
        _springer_bar_ax(ax)

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _bar_panel(ax1, final_knowledge, COLORS["primary"],
               "Mean knowledge score", "Final Knowledge Score")

    ax2 = fig.add_subplot(gs[0, 1])
    _bar_panel(ax2, final_visited, COLORS["accent"],
               "Mean visited locations", "Exploration Coverage")

    ax3 = fig.add_subplot(gs[0, 2])
    _bar_panel(ax3, final_resolves, COLORS["secondary"],
               "Total Earth-proxy resolves", "Evidence Ingestion (Resolves)", fmt="{:.0f}")

    ax4 = fig.add_subplot(gs[1, 0])
    _bar_panel(ax4, final_reward, COLORS["warn"],
               "Mean reward (per-agent)", "Final Reward Level")

    elapsed = [(r.end_wall - r.start_wall) for r in results]
    ax5 = fig.add_subplot(gs[1, 1])
    _bar_panel(ax5, elapsed, COLORS["grey"],
               "Wall-clock duration (s)", "Scenario Run Duration", fmt="{:.0f}s")

    # 6. Knowledge growth overlay — all scenarios on one plot, sequential x
    ax6 = fig.add_subplot(gs[1, 2])
    scenario_colors = plt.cm.tab10(np.linspace(0, 0.8, n))
    for i, r in enumerate(results):
        agg = _aggregate_snapshots(r.snapshots)
        if len(agg) >= 2:
            xs = list(range(len(agg)))
            ys = [a["mean_knowledge"] for a in agg]
            ax6.plot(xs, ys, linewidth=1.8, color=scenario_colors[i], label=r.name)
    ax6.set_ylim(bottom=0)
    n_pts = max((len(_aggregate_snapshots(r.snapshots)) for r in results), default=1)
    _apply_axis_style(ax6, list(range(n_pts)),
                      xlabel="Snapshot index",
                      ylabel="Mean knowledge score",
                      title="Knowledge Growth — All Scenarios")
    ax6.legend(fontsize=7, framealpha=0.9)

    fig.suptitle("Cross-Scenario Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "cross_scenario_comparison.png")


def chart_regime_shift(results: list[ScenarioResult], out: Path) -> None:
    """Dedicated chart for step-change scenarios showing before/after behavior."""
    shift_results = [r for r in results if "step_change" in r.name]
    if not shift_results:
        return

    n_plots = len(shift_results)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 5 * n_plots), squeeze=False)

    for idx, r in enumerate(shift_results):
        agg = _aggregate_snapshots(r.snapshots)
        if len(agg) < 3:
            continue

        xs = list(range(len(agg)))
        knowledge = [a["mean_knowledge"] for a in agg]
        reward = [a["mean_reward"] for a in agg]

        # Estimate change point — first third is pre-shift
        mid = len(xs) // 3

        # Knowledge
        ax = axes[idx][0]
        ax.plot(xs[:mid + 1], knowledge[:mid + 1], linewidth=2, color=COLORS["primary"], label="Pre-shift")
        ax.plot(xs[mid:], knowledge[mid:], linewidth=2, color=COLORS["danger"], label="Post-shift")
        ax.axvline(xs[mid], color=COLORS["grey"], linestyle="--", linewidth=1, alpha=0.7, label="Regime shift")
        _apply_axis_style(ax, xs,
                          xlabel="Snapshot (collected in order)",
                          ylabel="Mean Knowledge Score",
                          title=f"{r.name}: Knowledge Across Regime Shift")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # Reward
        ax2 = axes[idx][1]
        ax2.plot(xs[:mid + 1], reward[:mid + 1], linewidth=2, color=COLORS["accent"], label="Pre-shift")
        ax2.plot(xs[mid:], reward[mid:], linewidth=2, color=COLORS["danger"], label="Post-shift")
        ax2.axvline(xs[mid], color=COLORS["grey"], linestyle="--", linewidth=1, alpha=0.7, label="Regime shift")
        _apply_axis_style(ax2, xs,
                          xlabel="Snapshot (collected in order)",
                          ylabel="Mean Reward (per-agent)",
                          title=f"{r.name}: Reward Across Regime Shift")
        ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle("Regime Shift Adaptation Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "regime_shift_analysis.png")


# ============================= Data Export ==================================

def save_scenario_data(result: ScenarioResult, out: Path) -> None:
    """Save raw data as JSON for reproducibility."""
    import csv

    # Tick-level ndjson
    ndjson_path = out / "ticks.ndjson"
    with ndjson_path.open("w", encoding="utf-8") as f:
        for t in result.ticks:
            f.write(json.dumps({
                "tick": t.tick,
                "wall_time": t.wall_time,
                "agent_events_count": len(t.agent_events),
                "earth_proxy_resolves": t.earth_proxy_resolves,
                "agent_phase_exceeded": t.agent_phase_exceeded,
                "weather_updated": t.weather_updated,
                "wind_updated": t.wind_updated,
                "atmosphere_updated": t.atmosphere_updated,
                "astronomy_updated": t.astronomy_updated,
                "data_feeds_updated": t.data_feeds_updated,
            }) + "\n")

    # Snapshot-level CSV
    agg = _aggregate_snapshots(result.snapshots)
    if agg:
        csv_path = out / "summary.csv"
        keys = ["tick", "n_agents", "mean_knowledge", "std_knowledge", "median_knowledge",
                "max_knowledge", "mean_visited", "std_visited", "max_visited",
                "mean_reward", "std_reward", "mean_energy",
                "unique_locations_occupied", "earth_proxy_resolves"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(agg)

    # Agent snapshot JSON (last snapshot)
    if result.snapshots:
        agents_path = out / "agents_final.json"
        with agents_path.open("w", encoding="utf-8") as f:
            json.dump(result.snapshots[-1].agents, f, indent=2, ensure_ascii=False)

    # Full snapshot series (ndjson) for exact chart regeneration/backfill
    if PERSIST_SNAPSHOTS and result.snapshots:
        snaps_path = out / "snapshots.ndjson"
        with snaps_path.open("w", encoding="utf-8") as f:
            for s in result.snapshots:
                f.write(
                    json.dumps(
                        {
                            "tick": s.tick,
                            "wall_time": s.wall_time,
                            "agents": s.agents,
                            "earth_proxy": s.earth_proxy,
                            "config": s.config,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ===================== TimescaleDB telemetry (S85) =========================

def _epoch_to_dt(epoch: float):
    """Epoch seconds -> tz-aware UTC datetime (TIMESTAMPTZ-friendly)."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc)


def _iso_to_dt(iso: str, fallback_epoch: float):
    """Best-effort ISO-8601 -> datetime; falls back to an epoch on parse error."""
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return _epoch_to_dt(fallback_epoch)


async def write_telemetry_to_hypertables(result: ScenarioResult, dsn: str) -> None:
    """S85 — write one scenario's tick-level telemetry into the TimescaleDB
    hypertables created by migration f2b8e6d04a19 (tick_metrics, agent_events,
    adapter_latencies). NDJSON/CSV export (save_scenario_data) is unchanged and
    remains the read-only artifact; this is an additional sink, opt-in via
    --db-url / EARTHLINK_EVAL_DB_URL.

    social_events is intentionally NOT written here: that data lives only inside
    the live server's social-learning step and is not exposed on
    /api/eval/snapshot, so eval_suite has no source for it (server-side
    follow-up). Writes degrade loudly — a connection/insert failure is logged
    and re-raised so a misconfigured DSN is visible, not silently swallowed.
    """
    import asyncpg  # lazy: only required when --db-url is used

    tick_rows = []
    adapter_rows = []
    for s in result.snapshots:
        ts = _epoch_to_dt(s.wall_time)
        ks = [a.get("knowledge_score") for a in s.agents if a.get("knowledge_score") is not None]
        es = [a.get("energy") for a in s.agents if a.get("energy") is not None]
        ep = s.earth_proxy or {}
        tick_rows.append((
            ts, int(s.tick), None, len(s.agents),
            None,  # location_count not in snapshot
            ep.get("total_resolves"),
            None,  # is_running not per-snapshot here
            (sum(ks) / len(ks)) if ks else None,
            (sum(es) / len(es)) if es else None,
        ))
        for name, st in (ep.get("adapters") or {}).items():
            adapter_rows.append((
                ts, str(name), st.get("avg_latency_ms"),
                st.get("calls"), st.get("timeouts"), st.get("rate_limited"),
            ))

    agent_rows = []
    for t in result.ticks:
        ts = _iso_to_dt(t.wall_time, time.time())
        for ev in t.agent_events:
            goal = ev.get("goal")
            agent_rows.append((
                ts, int(t.tick), str(ev.get("agent_id")),
                ev.get("action"), ev.get("from_location_id"), ev.get("to_location_id"),
                ev.get("moved"), ev.get("distance_km"), ev.get("knowledge_score"),
                ev.get("reward"), ev.get("q_value"), ev.get("energy"),
                json.dumps(goal) if goal is not None else None,
            ))

    conn = await asyncpg.connect(dsn)
    try:
        await conn.executemany(
            "INSERT INTO tick_metrics (time, tick_count, tick_wall_ms, agent_count, "
            "location_count, total_resolves, is_running, avg_knowledge, avg_energy) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
            tick_rows,
        )
        await conn.executemany(
            "INSERT INTO adapter_latencies (time, adapter_name, avg_latency_ms, "
            "call_count, timeouts, rate_limited) VALUES ($1,$2,$3,$4,$5,$6)",
            adapter_rows,
        )
        await conn.executemany(
            "INSERT INTO agent_events (time, tick_count, agent_id, action, "
            "from_location_id, to_location_id, moved, distance_km, knowledge_score, "
            "reward, q_value, energy, goal) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb)",
            agent_rows,
        )
    finally:
        await conn.close()

    print(
        f"  [S85] wrote telemetry to TimescaleDB: {len(tick_rows)} tick_metrics, "
        f"{len(agent_rows)} agent_events, {len(adapter_rows)} adapter_latencies"
    )


# ============================= Main =========================================

async def main() -> None:
    parser = argparse.ArgumentParser(description="EarthLink Empirical Evaluation Suite")
    parser.add_argument("--base-url", default=os.environ.get("EARTHLINK_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--ws-url", default=os.environ.get("EARTHLINK_WS_URL", "ws://localhost:8000/ws/world"))
    parser.add_argument("--out", default=None, help="Output directory (default: artifacts/<timestamp>)")
    parser.add_argument("--quick", action="store_true", help="Run shorter scenarios for quick smoke testing")
    parser.add_argument("--reset-between", action="store_true", help="Reset world between scenarios")
    parser.add_argument("--recv-timeout", type=float, default=30.0, help="WS recv timeout")
    parser.add_argument("--snapshot-every", type=int, default=3, help="Take REST snapshot every N ticks")
    parser.add_argument(
        "--snapshot-http-timeout",
        type=float,
        default=60.0,
        help="HTTP timeout (seconds) for /api/eval/snapshot polling.",
    )
    parser.add_argument(
        "--version-http-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout (seconds) for initial /api/version reachability check.",
    )
    parser.add_argument(
        "--version-retries",
        type=int,
        default=8,
        help="Retries for initial /api/version check before failing.",
    )
    parser.add_argument(
        "--version-retry-sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between /api/version retries.",
    )
    parser.add_argument(
        "--eps",
        action="store_true",
        help="Also save EPS versions of charts (adds .eps files alongside .png).",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save PDF versions of charts. PDF embeds Type 42 fonts correctly and renders in all viewers.",
    )
    parser.add_argument(
        "--eps-only-missing",
        action="store_true",
        help="When saving EPS, only write .eps files that don't already exist.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Do not write any PNG charts (useful for backfilling EPS without touching existing PNGs).",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Do not write any data files (ticks.ndjson/summary.csv/agents_final.json/snapshots.ndjson).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write REPORT.md.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Do not persist snapshots.ndjson (keeps artifacts smaller but prevents full chart backfill).",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("EARTHLINK_EVAL_DB_URL"),
        help="S85: if set, also write tick-level telemetry to the TimescaleDB "
             "hypertables (postgresql://user:pass@host:port/db). Off by default — "
             "NDJSON/CSV artifacts are unaffected. Requires the schema migrated "
             "(alembic upgrade head) and the asyncpg package.",
    )
    args = parser.parse_args()

    global SAVE_EPS
    SAVE_EPS = bool(args.eps or os.environ.get("EARTHLINK_SAVE_EPS") == "1")

    global SAVE_PDF
    SAVE_PDF = bool(args.pdf or os.environ.get("EARTHLINK_SAVE_PDF") == "1")

    global SAVE_PNG, EPS_ONLY_MISSING, WRITE_DATA, WRITE_REPORT, PERSIST_SNAPSHOTS
    SAVE_PNG = not bool(args.no_png)
    EPS_ONLY_MISSING = bool(args.eps_only_missing)
    WRITE_DATA = not bool(args.no_data)
    WRITE_REPORT = not bool(args.no_report)
    PERSIST_SNAPSHOTS = not bool(args.no_snapshots)

    global SNAPSHOT_HTTP_TIMEOUT_SECONDS
    SNAPSHOT_HTTP_TIMEOUT_SECONDS = float(args.snapshot_http_timeout)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out) if args.out else Path(__file__).parent / "artifacts" / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    if args.quick:
        suite = [
            Scenario(name="baseline_15", tick_interval_seconds=1.0, ticks=15),
            Scenario(name="fast_15", tick_interval_seconds=0.2, ticks=15),
            Scenario(name="slow_10", tick_interval_seconds=2.0, ticks=10),
            Scenario(
                name="step_change_20",
                tick_interval_seconds=1.0, ticks=20,
                config_change_at_tick=7, config_change_to_interval_seconds=0.2,
            ),
        ]
        snapshot_every = 2
    else:
        suite = [
            Scenario(name="baseline_60", tick_interval_seconds=1.0, ticks=60),
            Scenario(name="fast_60", tick_interval_seconds=0.2, ticks=60),
            Scenario(name="slow_40", tick_interval_seconds=2.0, ticks=40),
            Scenario(
                name="step_change_80",
                tick_interval_seconds=1.0, ticks=80,
                config_change_at_tick=25, config_change_to_interval_seconds=0.2,
            ),
            Scenario(
                name="step_change_back_80",
                tick_interval_seconds=0.2, ticks=80,
                config_change_at_tick=30, config_change_to_interval_seconds=1.0,
            ),
        ]
        snapshot_every = args.snapshot_every

    # Reachability check
    last_exc: Exception | None = None
    for attempt in range(1, int(args.version_retries) + 1):
        try:
            with httpx.Client(timeout=float(args.version_http_timeout)) as c:
                resp = c.get(f"{args.base_url}/api/version")
                _ensure_ok(resp)
                print(f"Server: {resp.json()}")
                last_exc = None
                break
        except Exception as exc:
            last_exc = exc
            if attempt < int(args.version_retries):
                await asyncio.sleep(float(args.version_retry_sleep))
                continue
    if last_exc is not None:
        raise RuntimeError(
            f"Server not reachable at {args.base_url}. Start earthlink-server first. Error: {last_exc}"
        )

    results: list[ScenarioResult] = []

    for scenario in suite:
        print(f"\n{'='*60}")
        print(f"==> {scenario.name}  (ticks={scenario.ticks}, interval={scenario.tick_interval_seconds}s)")
        print(f"{'='*60}")

        result = await collect_scenario(
            args.base_url, args.ws_url, scenario,
            reset_before=args.reset_between,
            snapshot_every_ticks=snapshot_every,
            recv_timeout=args.recv_timeout,
        )
        results.append(result)
        elapsed = result.end_wall - result.start_wall
        print(f"<== {scenario.name} done in {elapsed:.1f}s  ({len(result.ticks)} ticks, {len(result.snapshots)} snapshots)")

        # Per-scenario output
        scenario_dir = out_root / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Generating charts for {scenario.name}...")
        agg = _aggregate_snapshots(result.snapshots)
        agent_series = _agent_time_series(result.snapshots)
        suffix = f"  --  {scenario.name}"

        if WRITE_DATA:
            save_scenario_data(result, scenario_dir)
        if args.db_url:
            await write_telemetry_to_hypertables(result, args.db_url)
        chart_tick_cadence(result, scenario_dir)
        chart_knowledge_growth(agg, scenario_dir, suffix)
        chart_exploration_frontier(agg, agent_series, scenario_dir, suffix)
        chart_reward_trajectory(agg, scenario_dir, suffix)
        chart_action_distribution(agg, scenario_dir, suffix)
        chart_goal_distribution(agg, scenario_dir, suffix)
        chart_energy_distribution(result.snapshots, scenario_dir, suffix)
        chart_earth_proxy_throughput(agg, scenario_dir, suffix)
        chart_geographic_spread(result.snapshots, scenario_dir, suffix)
        chart_adapter_performance(result.snapshots, scenario_dir, suffix)
        chart_system_health(result, scenario_dir)

    # Cross-scenario comparison
    if len(results) >= 2:
        print(f"\n  Generating cross-scenario comparison charts...")
        chart_cross_scenario_comparison(results, out_root)
        chart_regime_shift(results, out_root)

    # Summary report
    report_path = out_root / "REPORT.md"
    if WRITE_REPORT:
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"# EarthLink Evaluation Report\n\n")
            f.write(f"**Date**: {stamp}\n")
            f.write(f"**Server**: {args.base_url}\n")
            f.write(f"**Mode**: {'quick' if args.quick else 'full'}\n\n")
            f.write(f"## Scenarios\n\n")
            f.write(f"| Scenario | Ticks | Snapshots | Duration (s) | Final Mean Knowledge | Final Mean Visited |\n")
            f.write(f"|----------|-------|-----------|--------------|---------------------|-------------------|\n")
            for r in results:
                agg = _aggregate_snapshots(r.snapshots)
                mk = f"{agg[-1]['mean_knowledge']:.2f}" if agg else "N/A"
                mv = f"{agg[-1]['mean_visited']:.1f}" if agg else "N/A"
                dur = f"{r.end_wall - r.start_wall:.1f}"
                f.write(f"| {r.name} | {len(r.ticks)} | {len(r.snapshots)} | {dur} | {mk} | {mv} |\n")
            f.write(f"\n## Charts\n\n")
            f.write(f"### Cross-Scenario\n")
            f.write(f"- `cross_scenario_comparison.png` -- Overview of all metrics\n")
            f.write(f"- `regime_shift_analysis.png` -- Before/after regime shift behavior\n\n")
            f.write(f"### Per-Scenario\n")
            for r in results:
                f.write(f"\n**{r.name}/**\n")
                for chart in ["tick_cadence", "knowledge_growth", "exploration_frontier",
                              "reward_trajectory", "action_distribution", "goal_distribution",
                              "energy_distribution", "earth_proxy_throughput", "geographic_spread",
                              "adapter_performance", "system_health"]:
                    f.write(f"- `{chart}.png`\n")

    print(f"\n{'='*60}")
    print(f"Artifacts: {out_root}")
    if WRITE_REPORT:
        print(f"Report:    {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
