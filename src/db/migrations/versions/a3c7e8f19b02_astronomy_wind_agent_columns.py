"""astronomy_wind_agent_columns

Add new astronomy columns (celestial mechanics expansion),
wind_data table, and agent_state table.

wind_data and agent_state may already exist (created by SQLAlchemy
create_all at startup), so we use raw SQL with IF NOT EXISTS.

Revision ID: a3c7e8f19b02
Revises: 60fb13540f71
Create Date: 2026-02-17 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a3c7e8f19b02'
down_revision: Union[str, None] = '60fb13540f71'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_column_if_not_exists(table: str, column: str, col_type: str) -> None:
    """Add a column only if it doesn't already exist (idempotent)."""
    op.execute(sa.text(
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
    ))


def upgrade() -> None:
    # --- Astronomy: add new columns from celestial mechanics expansion ---
    _add_column_if_not_exists('astronomy_data', 'solar_noon', 'TIMESTAMP')
    _add_column_if_not_exists('astronomy_data', 'civil_dawn', 'TIMESTAMP')
    _add_column_if_not_exists('astronomy_data', 'civil_dusk', 'TIMESTAMP')
    _add_column_if_not_exists('astronomy_data', 'nautical_dawn', 'TIMESTAMP')
    _add_column_if_not_exists('astronomy_data', 'nautical_dusk', 'TIMESTAMP')
    _add_column_if_not_exists('astronomy_data', 'moon_illumination_pct', 'DOUBLE PRECISION')
    _add_column_if_not_exists('astronomy_data', 'moon_age_days', 'DOUBLE PRECISION')
    _add_column_if_not_exists('astronomy_data', 'moonrise', 'TIMESTAMP')
    _add_column_if_not_exists('astronomy_data', 'moonset', 'TIMESTAMP')

    # --- Wind data table (may already exist via create_all) ---
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS wind_data (
            id SERIAL PRIMARY KEY,
            location_id INTEGER NOT NULL REFERENCES locations(id),
            datetime TIMESTAMP NOT NULL,
            wind_speed_10m DOUBLE PRECISION,
            wind_direction_10m DOUBLE PRECISION,
            wind_gusts_10m DOUBLE PRECISION,
            wind_speed_upper DOUBLE PRECISION,
            wind_direction_upper DOUBLE PRECISION,
            upper_height_m INTEGER,
            pressure_hpa DOUBLE PRECISION
        )
    """))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_wind_data_location_id ON wind_data (location_id)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_wind_data_datetime ON wind_data (datetime)"
    ))

    # --- Agent state table (may already exist via create_all) ---
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS agent_state (
            id VARCHAR(32) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            location_id INTEGER NOT NULL REFERENCES locations(id),
            energy DOUBLE PRECISION NOT NULL,
            last_move_distance_km DOUBLE PRECISION NOT NULL,
            last_action VARCHAR(255) NOT NULL,
            learning_rate DOUBLE PRECISION NOT NULL,
            exploration_bias DOUBLE PRECISION NOT NULL,
            risk_tolerance DOUBLE PRECISION NOT NULL,
            stamina DOUBLE PRECISION NOT NULL,
            comfort_temperature_c DOUBLE PRECISION NOT NULL,
            knowledge JSONB NOT NULL DEFAULT '{}'
        )
    """))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_agent_state_location_id ON agent_state (location_id)"
    ))


def downgrade() -> None:
    op.drop_table('agent_state')
    op.drop_table('wind_data')
    op.drop_column('astronomy_data', 'moonset')
    op.drop_column('astronomy_data', 'moonrise')
    op.drop_column('astronomy_data', 'moon_age_days')
    op.drop_column('astronomy_data', 'moon_illumination_pct')
    op.drop_column('astronomy_data', 'nautical_dusk')
    op.drop_column('astronomy_data', 'nautical_dawn')
    op.drop_column('astronomy_data', 'civil_dusk')
    op.drop_column('astronomy_data', 'civil_dawn')
    op.drop_column('astronomy_data', 'solar_noon')
