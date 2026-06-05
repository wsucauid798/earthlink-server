"""timescale_hypertables — telemetry hypertables (tick / agent / adapter / social)

S84 from server-design-plan.md — design the hypertable schema for tick-level
telemetry. S85 (separate) wires `eval_suite` / the live server to write here;
this migration only creates the tables, turns them into hypertables, and sets
time partitioning + retention. Requires the TimescaleDB extension (S83,
e1a4d7c93f52).

Column choices are grounded in the telemetry the system already produces:
  - tick_metrics      ← World._tick_loop heartbeat (tick_count, wall_ms) +
                        /api/eval/snapshot world counts.
  - agent_events      ← the per-agent tick event dict from AgentSystem
                        (_tick_sequential / actor.tick): action, knowledge,
                        reward, q_value, energy, movement.
  - adapter_latencies ← EarthProxy.get_stats() per-adapter avg_latency_ms /
                        call_count / timeouts / rate_limited.
  - social_events     ← AgentSystem social-learning events (encounter,
                        dialogue, teaching, learning).

Partitioning: 1-day chunks. At 1000 agents × ~1 tick/s, agent_events is the
hottest table (~1k rows/tick); 1-day chunks keep recent data in a few chunks
the planner can prune efficiently while staying coarse enough to avoid chunk
sprawl. Retention: high-volume per-entity tables (agent_events, social_events)
kept 90 days; low-volume rollups (tick_metrics, adapter_latencies) kept 1 year.

No foreign keys to `locations`: these are append-only analytics tables and the
locations table is large (~2.5M rows); a per-insert FK check on the tick hot
path is not worth the referential guarantee. location ids are stored raw.

Revision ID: f2b8e6d04a19
Revises: e1a4d7c93f52
Create Date: 2026-06-05 00:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f2b8e6d04a19'
down_revision: Union[str, None] = 'e1a4d7c93f52'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (table, chunk_interval, retention_interval)
_HYPERTABLES = [
    ("tick_metrics", "1 day", "365 days"),
    ("agent_events", "1 day", "90 days"),
    ("adapter_latencies", "1 day", "365 days"),
    ("social_events", "1 day", "90 days"),
]


def upgrade() -> None:
    # --- Tables ------------------------------------------------------------
    op.execute(sa.text(
        """
        CREATE TABLE IF NOT EXISTS tick_metrics (
            time            TIMESTAMPTZ      NOT NULL,
            tick_count      INTEGER          NOT NULL,
            tick_wall_ms    DOUBLE PRECISION NULL,
            agent_count     INTEGER          NULL,
            location_count  BIGINT           NULL,
            total_resolves  BIGINT           NULL,
            is_running      BOOLEAN          NULL,
            avg_knowledge   DOUBLE PRECISION NULL,
            avg_energy      DOUBLE PRECISION NULL
        )
        """
    ))
    op.execute(sa.text(
        """
        CREATE TABLE IF NOT EXISTS agent_events (
            time              TIMESTAMPTZ      NOT NULL,
            tick_count        INTEGER          NOT NULL,
            agent_id          VARCHAR(255)     NOT NULL,
            action            VARCHAR(64)      NULL,
            from_location_id  INTEGER          NULL,
            to_location_id    INTEGER          NULL,
            moved             BOOLEAN          NULL,
            distance_km       DOUBLE PRECISION NULL,
            knowledge_score   DOUBLE PRECISION NULL,
            reward            DOUBLE PRECISION NULL,
            q_value           DOUBLE PRECISION NULL,
            energy            DOUBLE PRECISION NULL,
            goal              JSONB            NULL
        )
        """
    ))
    op.execute(sa.text(
        """
        CREATE TABLE IF NOT EXISTS adapter_latencies (
            time            TIMESTAMPTZ      NOT NULL,
            adapter_name    VARCHAR(128)     NOT NULL,
            avg_latency_ms  DOUBLE PRECISION NULL,
            call_count      BIGINT           NULL,
            timeouts        BIGINT           NULL,
            rate_limited    BIGINT           NULL
        )
        """
    ))
    op.execute(sa.text(
        """
        CREATE TABLE IF NOT EXISTS social_events (
            time         TIMESTAMPTZ  NOT NULL,
            tick_count   INTEGER      NULL,
            agent_id     VARCHAR(255) NOT NULL,
            peer_id      VARCHAR(255) NULL,
            event_type   VARCHAR(64)  NOT NULL,
            location_id  INTEGER      NULL,
            details      JSONB        NULL
        )
        """
    ))

    # --- Hypertables + retention ------------------------------------------
    for table, chunk_interval, retention in _HYPERTABLES:
        op.execute(sa.text(
            f"SELECT create_hypertable('{table}', by_range('time', INTERVAL '{chunk_interval}'), "
            f"if_not_exists => TRUE)"
        ))
        op.execute(sa.text(
            f"SELECT add_retention_policy('{table}', INTERVAL '{retention}', if_not_exists => TRUE)"
        ))

    # --- Secondary indexes (time index is auto-created by create_hypertable)
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS agent_events_agent_time_idx "
        "ON agent_events (agent_id, time DESC)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS social_events_agent_time_idx "
        "ON social_events (agent_id, time DESC)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS adapter_latencies_name_time_idx "
        "ON adapter_latencies (adapter_name, time DESC)"
    ))


def downgrade() -> None:
    # Retention policies are dropped automatically with their hypertable.
    for table, _chunk, _retention in _HYPERTABLES:
        op.execute(sa.text(f"DROP TABLE IF EXISTS {table}"))
