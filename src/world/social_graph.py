"""S88 — Apache AGE social-graph analysis layer.

Read-only Cypher analytics over the `earthlink_social` graph (S87) plus an
on-demand populate from live agent social state. Deliberately standalone — none
of this runs in the per-tick/save loop; the populate is triggered explicitly
(API/tool) so graph maintenance never sits on the simulation hot path.

AGE notes baked in from probing the real engine (release/PG17/1.7.0):
  - Queries run over an asyncpg connection that has `LOAD 'age'` +
    `search_path = ag_catalog,...` set (AGE's internal DDL/functions need it).
  - asyncpg returns `agtype` columns as JSON-ish strings; vertices/edges carry a
    `::vertex` / `::edge` annotation that must be stripped before json.loads.
  - Supported: MERGE/SET, count(), variable-length `-[:T*1..n]->`, length(),
    OPTIONAL MATCH, UNWIND. NOT supported: list comprehension over path nodes
    (so paths are returned via nodes(p) and parsed here).
  - AGE has no built-in community detection, so communities are computed here
    (union-find over INTERACTED_WITH edges) — a connected-components
    approximation, not modularity-based clustering. Labelled as such in the API.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from config import settings

logger = logging.getLogger(__name__)

GRAPH = "earthlink_social"


def _dsn() -> str:
    # config.database_url is the SQLAlchemy form (postgresql+asyncpg://...);
    # asyncpg.connect wants the plain libpq form.
    return settings.database_url.replace("+asyncpg", "")


async def _connect() -> asyncpg.Connection:
    conn = await asyncpg.connect(_dsn())
    await conn.execute("LOAD 'age'")
    await conn.execute('SET search_path = ag_catalog, "$user", public')
    return conn


def _parse(agt: str | None) -> Any:
    """Parse an AGE agtype text value into Python (strips ::vertex/::edge tags)."""
    if agt is None:
        return None
    s = agt.replace("::vertex", "").replace("::edge", "").replace("::path", "")
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return agt


def _q(value: Any) -> str:
    """Escape a value for embedding in a single-quoted Cypher string literal."""
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


async def _cypher(conn: asyncpg.Connection, query: str, cols: str) -> list[asyncpg.Record]:
    return await conn.fetch(f"SELECT * FROM cypher('{GRAPH}', $$ {query} $$) AS {cols}")


# --------------------------------------------------------------------------
# Populate (on-demand; NOT on the tick path)
# --------------------------------------------------------------------------

async def populate_from_agents(agents: list) -> dict:
    """Sync the graph from live agent social state.

    Maps each agent's `social_memory` to INTERACTED_WITH edges (with
    encounter_count / trust) and `teaching_events` to TAUGHT edges (+ the
    LEARNED_FROM inverse). Idempotent via MERGE. Returns row counts.
    """
    conn = await _connect()
    agents_n = edges_i = edges_t = 0
    try:
        for ag in agents:
            aid = _q(getattr(ag, "agent_id", ""))
            name = _q(getattr(ag, "name", ""))
            await conn.execute(
                f"SELECT * FROM cypher('{GRAPH}', $$ "
                f"MERGE (a:Agent {{agent_id:'{aid}'}}) SET a.name='{name}' "
                f"$$) AS (a agtype)"
            )
            agents_n += 1

            knowledge = getattr(ag, "knowledge", None)
            if knowledge is None:
                continue

            for peer_id, mem in (getattr(knowledge, "social_memory", {}) or {}).items():
                pid = _q(peer_id)
                encounters = int(mem.get("encounter_count", 0) or 0)
                trust = float(mem.get("trust", 0.0) or 0.0)
                await conn.execute(
                    f"SELECT * FROM cypher('{GRAPH}', $$ "
                    f"MERGE (a:Agent {{agent_id:'{aid}'}}) "
                    f"MERGE (b:Agent {{agent_id:'{pid}'}}) "
                    f"MERGE (a)-[r:INTERACTED_WITH]->(b) "
                    f"SET r.encounters={encounters}, r.trust={trust} "
                    f"$$) AS (r agtype)"
                )
                edges_i += 1

            for ev in (getattr(knowledge, "teaching_events", []) or []):
                learner = ev.get("taught_agent")
                if not learner:
                    continue
                lid = _q(learner)
                await conn.execute(
                    f"SELECT * FROM cypher('{GRAPH}', $$ "
                    f"MERGE (a:Agent {{agent_id:'{aid}'}}) "
                    f"MERGE (b:Agent {{agent_id:'{lid}'}}) "
                    f"MERGE (a)-[:TAUGHT]->(b) "
                    f"MERGE (b)-[:LEARNED_FROM]->(a) "
                    f"$$) AS (r agtype)"
                )
                edges_t += 1
        return {"agents": agents_n, "interacted_with": edges_i, "taught": edges_t}
    finally:
        await conn.close()


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

async def degree_centrality(limit: int = 20) -> list[dict]:
    """Degree centrality per agent (undirected edge count). Includes isolated
    agents (degree 0) via OPTIONAL MATCH."""
    conn = await _connect()
    try:
        rows = await _cypher(
            conn,
            "MATCH (a:Agent) OPTIONAL MATCH (a)-[r]-(:Agent) "
            "RETURN a.agent_id, count(r)",
            "(agent_id agtype, degree agtype)",
        )
        out = [
            {"agent_id": _parse(r["agent_id"]), "degree": _parse(r["degree"]) or 0}
            for r in rows
        ]
        out.sort(key=lambda x: x["degree"], reverse=True)
        return out[: int(limit)]
    finally:
        await conn.close()


async def belief_path(src: str, dst: str, max_hops: int = 6) -> dict | None:
    """Shortest belief-transmission path (via TAUGHT edges) from src to dst.

    AGE has no shortestPath() function, so we enumerate variable-length paths up
    to max_hops and take the minimum-length one. Returns {hops, agents} or None.
    """
    conn = await _connect()
    try:
        rows = await _cypher(
            conn,
            f"MATCH p=(a:Agent {{agent_id:'{_q(src)}'}})-[:TAUGHT*1..{int(max_hops)}]->"
            f"(b:Agent {{agent_id:'{_q(dst)}'}}) "
            f"RETURN nodes(p), length(p) ORDER BY length(p) LIMIT 1",
            "(ns agtype, len agtype)",
        )
        if not rows:
            return None
        nodes = _parse(rows[0]["ns"]) or []
        agents = [n.get("properties", {}).get("agent_id") for n in nodes]
        return {"hops": _parse(rows[0]["len"]), "agents": agents}
    finally:
        await conn.close()


async def communities() -> list[dict]:
    """Connected-components over INTERACTED_WITH (a community-detection
    approximation — AGE has no modularity clustering). Returns clusters of
    agent_ids, largest first."""
    conn = await _connect()
    try:
        rows = await _cypher(
            conn,
            "MATCH (a:Agent)-[:INTERACTED_WITH]-(b:Agent) RETURN a.agent_id, b.agent_id",
            "(x agtype, y agtype)",
        )
        # union-find
        parent: dict[str, str] = {}

        def find(n: str) -> str:
            parent.setdefault(n, n)
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for r in rows:
            x, y = _parse(r["x"]), _parse(r["y"])
            if x is not None and y is not None:
                union(x, y)

        clusters: dict[str, list[str]] = {}
        for node in parent:
            clusters.setdefault(find(node), []).append(node)
        result = [
            {"size": len(members), "members": sorted(members)}
            for members in clusters.values()
        ]
        result.sort(key=lambda c: c["size"], reverse=True)
        return result
    finally:
        await conn.close()
