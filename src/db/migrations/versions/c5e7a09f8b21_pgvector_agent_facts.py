"""pgvector_agent_facts — install vector extension + agent_facts table

S71/S73/S74 from server-design-plan.md.

S71 — Embedding model + dimensions (decision frozen here so the schema
and any read/write path agree):
    Model:       BAAI/bge-m3
    Dimensions:  1024
    Notes:       Multilingual (relevant for global geography), state-of-the-art
                 retrieval quality, dense vectors usable directly with pgvector
                 HNSW. Served via TEI (Text Embeddings Inference). Pinning the
                 dimension at schema time means swapping to a different model
                 later requires a new migration + re-embedding.

S73 — Extension:
    CREATE EXTENSION IF NOT EXISTS vector;  (requires pgvector-bundled image)

S74 — agent_facts table:
    Per-agent semantic memory written by the learn() path. Each row is a
    single fact (location-tagged, tick-stamped) embedded once at write time.
    Read path queries by vector similarity within an agent's rows.

    HNSW chosen over IVFFlat: better recall at high QPS, no training step,
    incremental inserts work without rebuilding. Cost: slightly slower
    inserts. For our write rate (a few facts per agent per tick) that's
    well below the crossover point where IVFFlat would win.

Revision ID: c5e7a09f8b21
Revises: b4d8f2a31c07
Create Date: 2026-05-04 06:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c5e7a09f8b21'
down_revision: Union[str, None] = 'b4d8f2a31c07'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


EMBEDDING_DIMS = 1024  # MUST match the embedding model chosen in S71.


def upgrade() -> None:
    # --- Extension ---------------------------------------------------------
    # Idempotent. Requires a Postgres image that bundles pgvector (we use
    # pgvector/pgvector:pg17 in docker-compose.prod.yml).
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    # --- Table -------------------------------------------------------------
    op.execute(sa.text(
        f"""
        CREATE TABLE IF NOT EXISTS agent_facts (
            id           BIGSERIAL PRIMARY KEY,
            agent_id     VARCHAR(255) NOT NULL,
            content      TEXT         NOT NULL,
            embedding    vector({EMBEDDING_DIMS}) NOT NULL,
            location_id  INTEGER      NULL REFERENCES locations(id) ON DELETE SET NULL,
            tick_count   INTEGER      NULL,
            metadata     JSONB        NOT NULL DEFAULT '{{}}'::jsonb,
            created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
        )
        """
    ))

    # --- Indexes -----------------------------------------------------------
    # agent_id is the highest-selectivity filter (we always query within
    # one agent's facts). Vector similarity narrows results within that set.
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS agent_facts_agent_id_idx "
        "ON agent_facts (agent_id)"
    ))

    # Location filter for spatial questions ("what does the agent know
    # about this place"). Partial because many facts are not tied to a
    # specific location.
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS agent_facts_location_id_idx "
        "ON agent_facts (location_id) WHERE location_id IS NOT NULL"
    ))

    # HNSW vector index. Cosine distance because BGE / E5 family models
    # are trained for cosine similarity; their outputs are L2-normalised
    # so cosine and dot product give equivalent rankings, but cosine is
    # the canonical operator and the one the embedding-server samples
    # match.
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS agent_facts_embedding_hnsw_idx "
        "ON agent_facts USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    ))

    # Tick filter for time-travel queries ("what did this agent believe
    # at tick N?"). Cheap secondary index.
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS agent_facts_tick_idx "
        "ON agent_facts (tick_count) WHERE tick_count IS NOT NULL"
    ))


def downgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS agent_facts"))
    # Intentionally do NOT drop the extension — other schemas might land
    # on it in the future (belief_embeddings, social_memory_embeddings).
