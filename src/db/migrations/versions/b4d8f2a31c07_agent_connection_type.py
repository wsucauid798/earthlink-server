"""agent_connection_type

Track the connection type (road, rail, path, waterway, proximity) used
in the agent's last move so energy cost reflects actual terrain.

Revision ID: b4d8f2a31c07
Revises: a3c7e8f19b02
Create Date: 2026-03-30 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b4d8f2a31c07'
down_revision: Union[str, None] = 'a3c7e8f19b02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text(
        "ALTER TABLE agent_state "
        "ADD COLUMN IF NOT EXISTS last_move_connection_type VARCHAR(64) NOT NULL DEFAULT 'road'"
    ))


def downgrade() -> None:
    op.drop_column('agent_state', 'last_move_connection_type')
