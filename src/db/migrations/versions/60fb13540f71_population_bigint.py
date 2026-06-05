"""population_bigint — widen locations.population to BIGINT

Revision ID: 60fb13540f71
Revises: fdbc615ea0d5
Create Date: 2026-02-12 22:17:46.608247

REPAIRED 2026-06-05: this migration was originally auto-generated against an
empty database, so it emitted `create_table` for every table (locations,
world_state, astronomy_data, location_connections, weather_data) — duplicating
the initial migration (fdbc615ea0d5) and failing from-scratch with
`relation "locations" already exists`. That broke `alembic upgrade head` on any
fresh DB and contributed to a prod migration-recovery incident (prod was
stamped past this revision, so its broken DDL never actually executed anywhere).
Replaced with the single change its name always implied: widen
`locations.population` from INTEGER to BIGINT. Safe to edit in place precisely
because the original body never successfully ran.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '60fb13540f71'
down_revision: Union[str, None] = 'fdbc615ea0d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Widen locations.population INTEGER -> BIGINT (the migration's actual intent;
    # locations already exist from fdbc615ea0d5). Idempotent via USING cast.
    op.alter_column(
        'locations', 'population',
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=True,
        postgresql_using='population::bigint',
    )


def downgrade() -> None:
    op.alter_column(
        'locations', 'population',
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=True,
        postgresql_using='population::integer',
    )
