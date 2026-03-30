"""Backfill EPS charts for existing empirical-test artifacts.

Important:
- Does NOT convert PNG->EPS.
- Does NOT delete or modify any existing PNGs.
- Re-renders EPS natively using Matplotlib from saved data files.

Limitations:
Some eval_suite charts cannot be regenerated from the current saved data
(e.g., action/goal distributions, adapter_performance, energy_distribution,
full exploration_frontier with per-agent traces), because the underlying
per-agent/per-tick inputs were not persisted in the artifact folder.
This script will skip those and print what it could not reproduce.

Usage:
  python empirical-tests/backfill_eps.py empirical-tests/artifacts/20260306T183131Z
  python empirical-tests/backfill_eps.py empirical-tests/artifacts --recursive
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    # Prefer Type 42 fonts (TrueType) in both PS/EPS and PDF outputs.
    # EPS: selectable text, no bitmapped Type 3 (required by Springer).
    # PDF: renders correctly in all viewers; accepted by Springer pdflatex.
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _apply_axis_style(ax, xs: list, xlabel: str, ylabel: str, title: str) -> None:
    """Springer-style: clean white background, no grid, bottom/left spines only."""
    n = len(xs)
    ax.set_xlim(-0.3, n - 0.7)
    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    if (n - 1) not in ticks:
        ticks.append(n - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t + 1) for t in ticks])
    ax.set_xlabel(xlabel, fontsize=13, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=6)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=12, direction="out", length=4)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)


def _springer_bar_ax(ax) -> None:
    """Springer style for bar/scatter axes."""
    ax.grid(False)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=12, direction="out", length=4)


def _read_snapshots_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            rows.append(obj)
    return rows


def _agg_from_snapshots(snaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute the aggregate series used by eval_suite charts."""
    import math

    rows: list[dict[str, Any]] = []
    for s in snaps:
        tick = s.get("tick")
        agents = s.get("agents")
        if agents is None or not isinstance(agents, list):
            continue
        try:
            tick_int = int(tick)
        except Exception:
            continue

        ks: list[float] = []
        vl: list[float] = []
        rw: list[float] = []
        en: list[float] = []
        actions: dict[str, int] = {}
        goals: dict[str, int] = {}
        locations: set[int] = set()

        for a in agents:
            if not isinstance(a, dict):
                continue
            try:
                ks.append(float(a.get("knowledge_score", 0) or 0))
            except Exception:
                pass
            try:
                vl.append(float(a.get("visited_locations", 0) or 0))
            except Exception:
                pass
            try:
                rw.append(float(a.get("last_reward", 0) or 0))
            except Exception:
                pass
            try:
                en.append(float(a.get("energy", 0) or 0))
            except Exception:
                pass

            la = a.get("last_action") or "unknown"
            if not isinstance(la, str):
                la = str(la)
            actions[la] = actions.get(la, 0) + 1

            g = a.get("goal") or "none"
            if not isinstance(g, str):
                g = str(g)
            goals[g] = goals.get(g, 0) + 1

            lid = a.get("location_id")
            try:
                locations.add(int(lid))
            except Exception:
                pass

        if not ks:
            continue

        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        def _std(xs: list[float]) -> float:
            if not xs:
                return 0.0
            m = _mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

        earth_proxy = s.get("earth_proxy") if isinstance(s.get("earth_proxy"), dict) else {}
        rows.append(
            {
                "tick": tick_int,
                "n_agents": len(agents),
                "mean_knowledge": _mean(ks),
                "std_knowledge": _std(ks),
                "max_knowledge": max(ks),
                "mean_visited": _mean(vl),
                "std_visited": _std(vl),
                "max_visited": max(vl) if vl else 0.0,
                "mean_reward": _mean(rw),
                "std_reward": _std(rw),
                "mean_energy": _mean(en),
                "unique_locations_occupied": len(locations),
                "actions": actions,
                "goals": goals,
                "earth_proxy": earth_proxy,
            }
        )
    return rows


def render_missing_from_snapshots(
    snapshots_ndjson: Path,
    out_dir: Path | None = None,
    src_dir: Path | None = None,
) -> list[str]:
    """Render charts that require full snapshots (not just summary/ticks).

    Args:
        snapshots_ndjson: Source data file.
        out_dir: Directory to write EPS files; defaults to snapshots_ndjson.parent.
        src_dir: Directory checked for existing PNGs (to decide what to regenerate);
                 defaults to snapshots_ndjson.parent.
    """
    if out_dir is None:
        out_dir = snapshots_ndjson.parent
    if src_dir is None:
        src_dir = snapshots_ndjson.parent
    snaps = _read_snapshots_ndjson(snapshots_ndjson)
    if len(snaps) < 2:
        return []

    agg = _agg_from_snapshots(snaps)
    if len(agg) < 2:
        return []

    written: list[str] = []

    # action_distribution.eps
    if _src_has_chart(src_dir, "action_distribution") and not _out_has_chart(out_dir, "action_distribution"):
        all_actions: set[str] = set()
        for r in agg:
            all_actions.update((r.get("actions") or {}).keys())
        xs = list(range(len(agg)))
        all_actions_sorted = sorted(all_actions)
        stacks: dict[str, list[float]] = {a: [] for a in all_actions_sorted}
        for r in agg:
            acts: dict[str, int] = r.get("actions") or {}
            total = sum(acts.values())
            for a in all_actions_sorted:
                v = float(acts.get(a, 0))
                stacks[a].append(v / total * 100 if total > 0 else 0.0)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].stackplot(xs, *[stacks[a] for a in all_actions_sorted], labels=all_actions_sorted, alpha=0.8)
        _apply_axis_style(axes[0], xs,
                          xlabel="Snapshot (collected in order)",
                          ylabel="Share of agents (%)",
                          title="Action Distribution Over Time")
        axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)
        axes[0].set_ylim(0, 100)

        final = (agg[-1].get("actions") or {})
        acts_sorted = sorted(final.keys())
        vals = [final[a] for a in acts_sorted]
        axes[1].bar(acts_sorted, vals, edgecolor="white", linewidth=0.8)
        axes[1].set_xlabel("Action", fontsize=10, labelpad=6)
        axes[1].set_ylabel("Agent Count", fontsize=10, labelpad=6)
        axes[1].set_title("Final Action Breakdown", fontsize=11, fontweight="bold", pad=8)
        _springer_bar_ax(axes[1])
        fig.suptitle("Agent Actions", fontsize=13, fontweight="bold")
        fig.tight_layout()
        _save_eps(fig, out_dir / "action_distribution.eps")
        written.append("action_distribution.eps")

    # goal_distribution.eps
    if _src_has_chart(src_dir, "goal_distribution") and not _out_has_chart(out_dir, "goal_distribution"):
        all_goals: set[str] = set()
        for r in agg:
            all_goals.update((r.get("goals") or {}).keys())
        xs = list(range(len(agg)))
        all_goals_sorted = sorted(all_goals)
        stacks: dict[str, list[float]] = {g: [] for g in all_goals_sorted}
        for r in agg:
            goals: dict[str, int] = r.get("goals") or {}
            total = sum(goals.values())
            for g in all_goals_sorted:
                v = float(goals.get(g, 0))
                stacks[g].append(v / total * 100 if total > 0 else 0.0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.stackplot(xs, *[stacks[g] for g in all_goals_sorted], labels=all_goals_sorted, alpha=0.8)
        _apply_axis_style(ax, xs,
                          xlabel="Snapshot (collected in order)",
                          ylabel="Share of agents (%)",
                          title="Goal Distribution Over Time")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_ylim(0, 100)
        fig.tight_layout()
        _save_eps(fig, out_dir / "goal_distribution.eps")
        written.append("goal_distribution.eps")

    # energy_distribution.eps
    if _src_has_chart(src_dir, "energy_distribution") and not _out_has_chart(out_dir, "energy_distribution"):
        first_agents = snaps[0].get("agents") if isinstance(snaps[0].get("agents"), list) else []
        last_agents = snaps[-1].get("agents") if isinstance(snaps[-1].get("agents"), list) else []
        if first_agents and last_agents:
            e_first = [float(a.get("energy", 0) or 0) for a in first_agents if isinstance(a, dict)]
            e_last = [float(a.get("energy", 0) or 0) for a in last_agents if isinstance(a, dict)]
            fig, ax = plt.subplots(figsize=(8, 5))
            parts = ax.violinplot([e_first, e_last], positions=[1, 2], showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_alpha(0.5)
            ax.set_xticks([1, 2])
            ax.set_xticklabels([f"Snapshot 1\n(start)", f"Snapshot {len(snaps)}\n(end)"], fontsize=9)
            ax.set_ylabel("Energy", fontsize=10, labelpad=6)
            ax.set_xlabel("Scenario Position", fontsize=10, labelpad=6)
            ax.set_title("Agent Energy Distribution", fontsize=11, fontweight="bold", pad=8)
            _springer_bar_ax(ax)
            fig.tight_layout()
            _save_eps(fig, out_dir / "energy_distribution.eps")
            written.append("energy_distribution.eps")

    # exploration_frontier.eps
    if _src_has_chart(src_dir, "exploration_frontier") and not _out_has_chart(out_dir, "exploration_frontier"):
        # Agent time series for visited_locations
        agent_series: dict[str, list[dict[str, Any]]] = {}
        for s in snaps:
            tick = s.get("tick")
            agents = s.get("agents")
            if not isinstance(agents, list):
                continue
            try:
                tick_int = int(tick)
            except Exception:
                continue
            for a in agents:
                if not isinstance(a, dict):
                    continue
                aid = a.get("id") or a.get("name") or "unknown"
                aid = str(aid)
                try:
                    visited = int(a.get("visited_locations", 0) or 0)
                except Exception:
                    visited = 0
                agent_series.setdefault(aid, []).append({"tick": tick_int, "visited_locations": visited})

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        xs = list(range(len(agg)))
        mean_vis = [float(r.get("mean_visited", 0.0)) for r in agg]
        std_vis = [float(r.get("std_visited", 0.0)) for r in agg]
        upper = [m + s for m, s in zip(mean_vis, std_vis)]
        lower = [max(0.0, m - s) for m, s in zip(mean_vis, std_vis)]
        max_vis = [float(r.get("max_visited", 0.0)) for r in agg]

        axes[0].fill_between(xs, lower, upper, alpha=0.25)
        axes[0].plot(xs, mean_vis, linewidth=2, label="Mean (45 agents)")
        axes[0].plot(xs, max_vis, linewidth=1.2, linestyle="--", label="Max agent")
        axes[0].set_ylim(bottom=0)
        _apply_axis_style(axes[0], xs,
                          xlabel="Snapshot (collected in order)",
                          ylabel="Unique Locations Visited",
                          title="Aggregate Exploration Coverage")
        axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)

        sample_ids = sorted(agent_series.keys())[:10]
        n_pts = len(agg)
        for aid in sample_ids:
            pts = agent_series[aid]
            pt_xs = list(range(len(pts)))
            axes[1].plot(pt_xs, [p["visited_locations"] for p in pts], linewidth=0.9, alpha=0.75, label=aid)
        axes[1].set_ylim(bottom=0)
        _apply_axis_style(axes[1], list(range(n_pts)),
                          xlabel="Snapshot (collected in order)",
                          ylabel="Unique Locations Visited",
                          title="Per-Agent Traces (sample of 10)")
        if sample_ids:
            axes[1].legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)

        fig.suptitle("Exploration Frontier", fontsize=13, fontweight="bold")
        fig.tight_layout()
        _save_eps(fig, out_dir / "exploration_frontier.eps")
        written.append("exploration_frontier.eps")

    # adapter_performance.eps
    if _src_has_chart(src_dir, "adapter_performance") and not _out_has_chart(out_dir, "adapter_performance"):
        ep = snaps[-1].get("earth_proxy") if isinstance(snaps[-1].get("earth_proxy"), dict) else None
        if ep:
            adapter_stats = ep.get("adapters", {}) if isinstance(ep.get("adapters"), dict) else {}
            if not adapter_stats:
                adapter_stats = {k: v for k, v in ep.items() if isinstance(v, dict) and "calls" in v}
            if adapter_stats:
                sorted_adapters = sorted(adapter_stats.items(), key=lambda x: (x[1] or {}).get("calls", 0), reverse=True)[:15]
                if sorted_adapters:
                    names = [a[0] for a in sorted_adapters]
                    calls = [int((a[1] or {}).get("calls", 0) or 0) for a in sorted_adapters]
                    latency = [float((a[1] or {}).get("avg_latency_ms", 0) or 0) for a in sorted_adapters]
                    timeouts = [int((a[1] or {}).get("timeouts", 0) or 0) for a in sorted_adapters]

                    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                    y_pos = list(range(len(names)))

                    axes[0].barh(y_pos, calls, alpha=0.85, edgecolor="white")
                    axes[0].set_yticks(y_pos)
                    axes[0].set_yticklabels(names, fontsize=8)
                    axes[0].set_xlabel("Total Calls", fontsize=10, labelpad=6)
                    axes[0].set_title("Adapter Call Volume", fontsize=11, fontweight="bold", pad=8)
                    axes[0].invert_yaxis()
                    _springer_bar_ax(axes[0])

                    axes[1].barh(y_pos, latency, alpha=0.85, edgecolor="white")
                    axes[1].set_yticks(y_pos)
                    axes[1].set_yticklabels(names, fontsize=8)
                    axes[1].set_xlabel("Avg Latency (ms)", fontsize=10, labelpad=6)
                    axes[1].set_title("Adapter Latency", fontsize=11, fontweight="bold", pad=8)
                    axes[1].invert_yaxis()
                    _springer_bar_ax(axes[1])

                    axes[2].barh(y_pos, timeouts, alpha=0.85, edgecolor="white")
                    axes[2].set_yticks(y_pos)
                    axes[2].set_yticklabels(names, fontsize=8)
                    axes[2].set_xlabel("Timeout Count", fontsize=10, labelpad=6)
                    axes[2].set_title("Adapter Timeouts", fontsize=11, fontweight="bold", pad=8)
                    axes[2].invert_yaxis()
                    _springer_bar_ax(axes[2])

                    fig.suptitle("Earth Adapter Performance", fontsize=13, fontweight="bold")
                    fig.tight_layout()
                    _save_eps(fig, out_dir / "adapter_performance.eps")
                    written.append("adapter_performance.eps")

    return written


def _read_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for r in reader:
            row: dict[str, Any] = dict(r)
            # best-effort numeric coercion
            for k, v in list(row.items()):
                if v is None:
                    continue
                v = v.strip()
                if v == "":
                    continue
                if k in {"tick", "n_agents", "max_knowledge", "max_visited", "unique_locations_occupied", "earth_proxy_resolves"}:
                    try:
                        row[k] = int(float(v))
                    except Exception:
                        pass
                else:
                    # floats
                    try:
                        row[k] = float(v)
                    except Exception:
                        pass
            rows.append(row)
        return rows


def _read_ticks_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


_SAVE_FORMAT: str = "eps"  # "eps" or "pdf"; set by --pdf flag in main()
_FORCE_OVERWRITE: bool = False
_USE_SNAPSHOTS: bool = True


def _src_has_chart(src_dir: Path, stem: str) -> bool:
    """True if any rendered chart with this stem exists in the source dir."""
    return any((src_dir / f"{stem}{ext}").exists() for ext in (".png", ".eps", ".pdf"))


def _out_has_chart(out_dir: Path, stem: str) -> bool:
    """True if the target-format output already exists in out_dir."""
    if _FORCE_OVERWRITE:
        return False
    ext = ".pdf" if _SAVE_FORMAT == "pdf" else ".eps"
    return (out_dir / f"{stem}{ext}").exists()


def _save_eps(fig, eps_path: Path) -> None:
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = eps_path if _SAVE_FORMAT == "eps" else eps_path.with_suffix(".pdf")
    fig.savefig(out_path, format=_SAVE_FORMAT)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def render_from_summary(summary_csv: Path, out_dir: Path | None = None) -> list[str]:
    if out_dir is None:
        out_dir = summary_csv.parent
    agg = _read_summary_csv(summary_csv)
    if len(agg) < 2:
        return []

    written: list[str] = []

    # knowledge_growth.eps
    if {"tick", "mean_knowledge", "std_knowledge", "max_knowledge"}.issubset(agg[0].keys()):
        xs = list(range(len(agg)))
        mean = [float(r["mean_knowledge"]) for r in agg]
        std = [float(r.get("std_knowledge", 0.0)) for r in agg]
        maxk = [float(r.get("max_knowledge", 0.0)) for r in agg]
        upper = [m + s for m, s in zip(mean, std)]
        lower = [max(0.0, m - s) for m, s in zip(mean, std)]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.fill_between(xs, lower, upper, alpha=0.25, label="\u00b11 SD")
        ax.plot(xs, mean, linewidth=2, label="Mean (45 agents)")
        ax.plot(xs, maxk, linewidth=1.2, linestyle="--", label="Max agent")
        ax.set_ylim(bottom=0)
        _apply_axis_style(ax, xs,
                          xlabel="Snapshot index",
                          ylabel="Knowledge score",
                          title="Agent Knowledge Growth")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        _save_eps(fig, out_dir / "knowledge_growth.eps")
        written.append("knowledge_growth.eps")

    # reward_trajectory.eps
    if {"tick", "mean_reward"}.issubset(agg[0].keys()):
        xs = list(range(len(agg)))
        mean_r = [float(r["mean_reward"]) for r in agg]
        std_r = [float(r.get("std_reward", 0.0)) for r in agg]
        upper = [m + s for m, s in zip(mean_r, std_r)]
        lower = [m - s for m, s in zip(mean_r, std_r)]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.fill_between(xs, lower, upper, alpha=0.25, label="\u00b11 SD")
        ax.plot(xs, mean_r, linewidth=2, label="Mean reward")
        ax.axhline(0, linewidth=0.8, linestyle=":", label="Zero baseline")
        _apply_axis_style(ax, xs,
                          xlabel="Snapshot index",
                          ylabel="Mean Reward (per-agent, per-tick)",
                          title="Reward Trajectory")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        _save_eps(fig, out_dir / "reward_trajectory.eps")
        written.append("reward_trajectory.eps")

    # earth_proxy_throughput.eps (total resolves over time)
    if {"tick", "earth_proxy_resolves"}.issubset(agg[0].keys()):
        xs = list(range(len(agg)))
        resolves = [float(r["earth_proxy_resolves"]) for r in agg]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, resolves, linewidth=2, label="Cumulative resolves")
        ax.fill_between(xs, 0, resolves, alpha=0.15)
        ax.set_ylim(bottom=0)
        _apply_axis_style(ax, xs,
                          xlabel="Snapshot index",
                          ylabel="Total Earth-proxy resolves (cumulative)",
                          title="Earth Proxy Evidence Throughput")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        _save_eps(fig, out_dir / "earth_proxy_throughput.eps")
        written.append("earth_proxy_throughput.eps")

    # geographic_spread.eps
    if {"tick", "unique_locations_occupied"}.issubset(agg[0].keys()):
        xs = list(range(len(agg)))
        unique = [float(r["unique_locations_occupied"]) for r in agg]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, unique, linewidth=2, marker="o", markersize=4, label="Occupied locations")
        ax.set_ylim(bottom=0)
        _apply_axis_style(ax, xs,
                          xlabel="Snapshot (collected in order)",
                          ylabel="Distinct occupied locations (count)",
                          title="Geographic Spread of Agents")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        _save_eps(fig, out_dir / "geographic_spread.eps")
        written.append("geographic_spread.eps")

    return written


def render_from_ticks(ticks_ndjson: Path, out_dir: Path | None = None) -> list[str]:
    if out_dir is None:
        out_dir = ticks_ndjson.parent
    ticks = _read_ticks_ndjson(ticks_ndjson)
    if len(ticks) < 3:
        return []

    written: list[str] = []

    # tick_cadence.eps
    intervals: list[float] = []
    for i in range(1, len(ticks)):
        # eval_suite ticks.ndjson stores wall_time as ISO string
        t1 = ticks[i - 1].get("wall_time")
        t2 = ticks[i].get("wall_time")
        if not (isinstance(t1, str) and isinstance(t2, str)):
            continue
        try:
            # fromisoformat handles timezone offsets
            from datetime import datetime

            d1 = datetime.fromisoformat(t1)
            d2 = datetime.fromisoformat(t2)
            intervals.append((d2 - d1).total_seconds())
        except Exception:
            continue

    if intervals:
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})
        med = float(np.median(intervals))
        xs = list(range(len(intervals)))

        axes[0].plot(xs, intervals, linewidth=1.2, alpha=0.8)
        axes[0].axhline(y=med, linestyle="--", linewidth=1, label=f"Median = {med:.2f}s")
        _apply_axis_style(axes[0], xs,
                          xlabel="Interval index (consecutive ticks)",
                          ylabel="Wall-clock interval (s)",
                          title="Tick Cadence Over Time")
        axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)

        axes[1].hist(intervals, bins=min(25, len(intervals)), alpha=0.8, edgecolor="white")
        axes[1].axvline(med, linestyle="--", linewidth=1.5)
        axes[1].set_xlabel("Interval (s)", fontsize=10, labelpad=6)
        axes[1].set_ylabel("Count", fontsize=10, labelpad=6)
        axes[1].set_title("Distribution", fontsize=11, fontweight="bold", pad=8)
        _springer_bar_ax(axes[1])

        fig.suptitle("Tick Cadence", fontsize=13, fontweight="bold")
        fig.tight_layout()
        _save_eps(fig, out_dir / "tick_cadence.eps")
        written.append("tick_cadence.eps")

    # system_health.eps
    xs_tick = []
    exceeded = []
    domains = ["weather", "wind", "atmosphere", "astronomy", "data_feeds"]
    domain_keys = [f"{d}_updated" for d in domains]
    updates_by_domain = {d: [] for d in domains}

    for t in ticks:
        tick = t.get("tick")
        if not isinstance(tick, int):
            try:
                tick = int(tick)
            except Exception:
                continue
        xs_tick.append(tick)
        exceeded.append(1 if t.get("agent_phase_exceeded") else 0)
        for d, k in zip(domains, domain_keys):
            updates_by_domain[d].append(1 if t.get(k) else 0)

    if xs_tick:
        xs_seq = list(range(len(xs_tick)))
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        axes[0].fill_between(xs_seq, 0, exceeded, step="mid", alpha=0.45, label="Phase exceeded")
        axes[0].set_ylabel("Agent phase\nbudget", fontsize=10)
        axes[0].set_yticks([0, 1])
        axes[0].set_yticklabels(["OK", "Exceeded"], fontsize=9)
        axes[0].set_ylim(-0.1, 1.4)
        axes[0].set_title("Agent Phase Budget Exceedance", fontsize=11, fontweight="bold", pad=6)
        _springer_bar_ax(axes[0])

        for i, d in enumerate(domains):
            axes[1].scatter(
                [x for x, u in zip(xs_seq, updates_by_domain[d]) if u],
                [i + 1] * sum(updates_by_domain[d]),
                s=14, label=d, alpha=0.85,
            )
        step = max(1, len(xs_seq) // 8)
        ticks = list(range(0, len(xs_seq), step))
        if (len(xs_seq) - 1) not in ticks:
            ticks.append(len(xs_seq) - 1)
        axes[1].set_xticks(ticks)
        axes[1].set_xticklabels([str(t + 1) for t in ticks], fontsize=9)
        axes[1].set_xlabel("Snapshot index", fontsize=10, labelpad=6)
        axes[1].set_ylabel("Domain", fontsize=10, labelpad=6)
        axes[1].set_yticks(range(1, len(domains) + 1))
        axes[1].set_yticklabels(domains, fontsize=9)
        axes[1].set_xlim(-0.5, len(xs_seq) - 0.5)
        axes[1].set_title("Data Refresh Events by Domain", fontsize=11, fontweight="bold", pad=6)
        axes[1].legend(loc="upper right", fontsize=8, framealpha=0.9)
        _springer_bar_ax(axes[1])

        fig.suptitle("System Health", fontsize=12, fontweight="bold")
        fig.tight_layout()
        _save_eps(fig, out_dir / "system_health.eps")
        written.append("system_health.eps")

    return written


def backfill_one_dir(run_dir: Path, out_root: Path | None = None) -> dict[str, Any]:
    """Backfill EPS for a single artifacts run dir.

    Args:
        run_dir:  Source artifacts directory (contains per-scenario subdirs + PNG charts).
        out_root: Root directory for EPS output.  Mirrors the structure of run_dir.
                  If None, EPS files are written alongside the source data files (old behaviour).
    """
    written: list[str] = []
    skipped: list[str] = []

    # Helper: resolve output path for a file inside a scenario or run-level dir.
    def _out_dir_for(src_subdir: Path) -> Path:
        if out_root is None:
            return src_subdir
        # Preserve relative structure under run_dir inside out_root.
        try:
            rel = src_subdir.relative_to(run_dir)
            return out_root / rel
        except ValueError:
            return src_subdir

    run_out = out_root if out_root is not None else run_dir

    def _regime_shift_from_summaries() -> bool:
        """Regenerate eval_suite's regime_shift_analysis chart from summary.csv.

        The original chart uses snapshot aggregates; summary.csv contains the
        same aggregate series (mean_knowledge/mean_reward over ticks), so we can
        reproduce the plot without touching PNGs.
        """
        shift_series: list[tuple[str, list[dict[str, Any]]]] = []
        for scenario_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
            if "step_change" not in scenario_dir.name:
                continue
            summary = scenario_dir / "summary.csv"
            if not summary.exists():
                continue
            agg = _read_summary_csv(summary)
            if len(agg) < 3:
                continue
            shift_series.append((scenario_dir.name, agg))

        if not shift_series:
            return False

        n_plots = len(shift_series)
        fig, axes = plt.subplots(n_plots, 2, figsize=(14, 5 * n_plots), squeeze=False)

        for idx, (name, agg) in enumerate(shift_series):
            xs = list(range(len(agg)))
            knowledge = [float(a.get("mean_knowledge", 0.0)) for a in agg]
            reward = [float(a.get("mean_reward", 0.0)) for a in agg]

            mid = len(xs) // 3  # match eval_suite heuristic

            ax = axes[idx][0]
            ax.plot(xs[: mid + 1], knowledge[: mid + 1], linewidth=2, label="Pre-shift")
            ax.plot(xs[mid:], knowledge[mid:], linewidth=2, label="Post-shift")
            ax.axvline(xs[mid], linestyle="--", linewidth=1, alpha=0.7, label="Regime shift")
            _apply_axis_style(ax, xs,
                              xlabel="Snapshot index",
                              ylabel="Mean knowledge score",
                              title=f"{name}: Knowledge Across Regime Shift")
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

            ax2 = axes[idx][1]
            ax2.plot(xs[: mid + 1], reward[: mid + 1], linewidth=2, label="Pre-shift")
            ax2.plot(xs[mid:], reward[mid:], linewidth=2, label="Post-shift")
            ax2.axvline(xs[mid], linestyle="--", linewidth=1, alpha=0.7, label="Regime shift")
            _apply_axis_style(ax2, xs,
                              xlabel="Snapshot index",
                              ylabel="Mean Reward (per-agent)",
                              title=f"{name}: Reward Across Regime Shift")
            ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

        fig.suptitle("Regime Shift Adaptation Analysis", fontsize=14, fontweight="bold")
        fig.tight_layout()
        _save_eps(fig, run_out / "regime_shift_analysis.eps")
        return True

    # per-scenario folders
    for scenario_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        scenario_out = _out_dir_for(scenario_dir)
        summary = scenario_dir / "summary.csv"
        ticks = scenario_dir / "ticks.ndjson"
        if summary.exists():
            written.extend([str(scenario_out / n) for n in render_from_summary(summary, out_dir=scenario_out)])
        if ticks.exists():
            written.extend([str(scenario_out / n) for n in render_from_ticks(ticks, out_dir=scenario_out)])

        # Charts that require full snapshots
        snaps = scenario_dir / "snapshots.ndjson"
        if _USE_SNAPSHOTS and snaps.exists():
            written.extend([
                str(scenario_out / n)
                for n in render_missing_from_snapshots(snaps, out_dir=scenario_out, src_dir=scenario_dir)
            ])

        # Known non-regenerable charts given current saved data
        for stem in [
            "action_distribution",
            "goal_distribution",
            "energy_distribution",
            "exploration_frontier",
            "adapter_performance",
        ]:
            if _src_has_chart(scenario_dir, stem) and not _out_has_chart(scenario_out, stem):
                ext = ".pdf" if _SAVE_FORMAT == "pdf" else ".eps"
                skipped.append(str(scenario_out / f"{stem}{ext}"))

    # run-level charts
    # These require combining scenario summary.csv files; we can do that.
    # If the PNG exists in the source run_dir, we will create EPS in run_out.
    if _src_has_chart(run_dir, "cross_scenario_comparison") and not _out_has_chart(run_out, "cross_scenario_comparison"):
        # Minimal reconstruction: bar charts from final snapshot rows in each scenario summary.csv
        scenario_summaries = []
        for scenario_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
            summary = scenario_dir / "summary.csv"
            if summary.exists():
                agg = _read_summary_csv(summary)
                if agg:
                    scenario_summaries.append((scenario_dir.name, agg[-1]))

        if len(scenario_summaries) >= 2:
            snames = [n for n, _ in scenario_summaries]
            final_knowledge = [float(r.get("mean_knowledge", 0.0)) for _, r in scenario_summaries]
            final_visited = [float(r.get("mean_visited", 0.0)) for _, r in scenario_summaries]
            final_reward = [float(r.get("mean_reward", 0.0)) for _, r in scenario_summaries]
            final_resolves = [float(r.get("earth_proxy_resolves", 0.0)) for _, r in scenario_summaries]

            def _bar_panel(ax, vals, ylabel, title):
                x = list(range(len(snames)))
                bars = ax.bar(x, vals, alpha=0.85, edgecolor="white", linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(snames, rotation=28, ha="right", fontsize=12)
                ax.set_ylabel(ylabel, fontsize=13, labelpad=5)
                ax.set_title(title, fontsize=14, fontweight="bold", pad=6)
                ax.set_ylim(bottom=0, top=max(vals) * 1.18 if vals else 1)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=12)
                _springer_bar_ax(ax)

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()

            _bar_panel(axes[0], final_knowledge, "Mean knowledge score", "Final Knowledge Score")
            _bar_panel(axes[1], final_visited, "Mean visited locations (count)", "Exploration Coverage")
            _bar_panel(axes[2], final_resolves, "Total Earth-proxy resolves (count)", "Evidence Ingestion (Resolves)")
            _bar_panel(axes[3], final_reward, "Mean reward (per-agent)", "Final Reward Level")

            fig.suptitle("Cross-Scenario Comparison", fontsize=16, fontweight="bold")
            fig.tight_layout()
            _save_eps(fig, run_out / "cross_scenario_comparison.eps")
            written.append(str(run_out / "cross_scenario_comparison.eps"))
        else:
            skipped.append(str(run_out / "cross_scenario_comparison.eps"))

    if _src_has_chart(run_dir, "regime_shift_analysis") and not _out_has_chart(run_out, "regime_shift_analysis"):
        if _regime_shift_from_summaries():
            written.append(str(run_out / "regime_shift_analysis.eps"))
        else:
            skipped.append(str(run_out / "regime_shift_analysis.eps"))

    return {"written": written, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate EPS charts from existing empirical-test artifact data files."
    )
    parser.add_argument("path", type=str, help="Run dir under empirical-tests/artifacts, or artifacts root")
    parser.add_argument("--recursive", action="store_true", help="Treat path as artifacts root and process all runs")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Root directory for output.  The directory structure of each run "
            "is mirrored under DIR so output files never mix with existing PNGs.  "
            "If omitted, files are written alongside the source data files."
        ),
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help=(
            "Save PDF instead of EPS.  PDF with pdf.fonttype=42 embeds fonts correctly "
            "and renders in all viewers (Acrobat, browsers, Windows).  "
            "Springer pdflatex accepts PDF figures via \\includegraphics."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target-format outputs (useful when updating labels).",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Skip snapshot-heavy charts (faster; still regenerates summary/tick-based figures).",
    )
    args = parser.parse_args()

    global _SAVE_FORMAT
    if args.pdf:
        _SAVE_FORMAT = "pdf"

    global _FORCE_OVERWRITE
    _FORCE_OVERWRITE = bool(args.force)

    global _USE_SNAPSHOTS
    _USE_SNAPSHOTS = not bool(args.no_snapshots)

    base = Path(args.path)
    if args.recursive:
        runs = sorted([p for p in base.iterdir() if p.is_dir()])
    else:
        runs = [base]

    out_root_base = Path(args.out_dir) if args.out_dir else None

    for run in runs:
        # When --out-dir is set, mirror run subdirectory name inside the output root.
        if out_root_base is not None:
            run_out = out_root_base / run.name
        else:
            run_out = None

        print(f"==> {run}")
        if run_out:
            print(f"    output -> {run_out}")
        res = backfill_one_dir(run, out_root=run_out)
        for w in res["written"]:
            print(f"  wrote {w}")
        if res["skipped"]:
            print("  skipped (missing inputs):")
            for s in res["skipped"]:
                print(f"    {s}")


if __name__ == "__main__":
    main()
