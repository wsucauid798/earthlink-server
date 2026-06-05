"""timescaledb_extension — install the TimescaleDB extension

S83 from server-design-plan.md.

S83 — Extension:
    CREATE EXTENSION IF NOT EXISTS timescaledb;

    Requires a Postgres image that bundles TimescaleDB *and* has
    `shared_preload_libraries = 'timescaledb'` set (TimescaleDB refuses to
    create the extension otherwise). We get both from the
    `timescale/timescaledb-ha:pg17` image, which also bundles pgvector — so the
    existing `CREATE EXTENSION vector` (c5e7a09f8b21) and this one coexist on a
    single image. See docker-compose.yml / vps-setup/docker-compose.prod.yml.

    No hypertables are created here — that is S84 (schema design) + S85
    (telemetry migration). This migration only makes the extension available,
    mirroring how c5e7a09f8b21 installed `vector` before any vector columns.

Revision ID: e1a4d7c93f52
Revises: c5e7a09f8b21
Create Date: 2026-06-05 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e1a4d7c93f52'
down_revision: Union[str, None] = 'c5e7a09f8b21'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Idempotent. Requires a Postgres image with TimescaleDB bundled and
    # preloaded (timescale/timescaledb-ha:pg17). On an image without it this
    # raises a clear error rather than silently degrading — by design, since a
    # missing time-series engine is a deployment fault, not a runtime one.
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb"))


def downgrade() -> None:
    # Intentionally do NOT drop the extension. Dropping it would CASCADE-drop
    # every hypertable built on it (S84/S85), which is never what a single
    # schema rollback intends. The extension is harmless when unused.
    pass
