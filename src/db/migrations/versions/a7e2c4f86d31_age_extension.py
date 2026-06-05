"""age_extension — install the Apache AGE extension

S86 from server-design-plan.md.

S86 — Extension:
    CREATE EXTENSION IF NOT EXISTS age;

    Requires a Postgres image with Apache AGE installed (vps-setup/Dockerfile.db
    compiles it on top of timescale/timescaledb-ha:pg17). `CREATE EXTENSION age`
    loads the library on demand, so AGE does not need to be in
    shared_preload_libraries for the extension to install. Sessions that *query*
    the graph must still `LOAD 'age'` and put `ag_catalog` on their search_path
    (handled in the S87 migration and the S88 query layer).

    No graph is created here — that is S87 (create_graph + vertex/edge labels).
    This mirrors how the vector (c5e7a09f8b21) and timescaledb (e1a4d7c93f52)
    extensions were installed before any objects that use them.

Revision ID: a7e2c4f86d31
Revises: f2b8e6d04a19
Create Date: 2026-06-05 01:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a7e2c4f86d31'
down_revision: Union[str, None] = 'f2b8e6d04a19'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Idempotent. Requires the AGE library present in PostgreSQL's lib dir
    # (vps-setup/Dockerfile.db). On an image without it this raises a clear
    # error rather than degrading — a missing graph engine is a deploy fault.
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS age"))


def downgrade() -> None:
    # Leave the extension in place (consistent with the vector/timescaledb
    # migrations). The S87 downgrade drops the graph; dropping the extension
    # itself is not part of a single schema rollback.
    pass
