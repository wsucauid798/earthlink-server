"""age_social_graph — create the AGE graph + vertex/edge labels

S87 from server-design-plan.md — the property-graph schema for social /
teaching / learning analysis. Requires the AGE extension (S86, a7e2c4f86d31).

Graph: `earthlink_social`
Vertices (vlabels):
    Agent     — one per AutonomousAgent (agent_id, name)
    Location  — places agents meet / visit (location_id, name)
Edges (elabels):
    INTERACTED_WITH  — Agent → Agent co-location / conversation
    TAUGHT           — Agent → Agent knowledge transfer (teacher → learner)
    LEARNED_FROM     — Agent → Agent (inverse view of teaching)
    BELIEVES_ABOUT   — Agent → Location belief edges
    VISITED          — Agent → Location movement history

These mirror the social-learning state the AgentSystem already tracks
(interaction_network, teaching_events, social_memory, visit history). The S88
query layer / write path populates them; this migration only declares the
graph and its labels so Cypher `MATCH`/`MERGE` have stable targets.

AGE specifics: the session must `LOAD 'age'` AND put `ag_catalog` on its
search_path before `create_graph`. Schema-qualifying the call is not enough —
`create_graph` internally emits index DDL referencing the `graphid_ops` operator
class unqualified, which only resolves with `ag_catalog` on the search_path
(otherwise: `operator class "graphid_ops" does not exist for access method
"btree"`).

Revision ID: b9f5d1a37e62
Revises: a7e2c4f86d31
Create Date: 2026-06-05 01:15:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b9f5d1a37e62'
down_revision: Union[str, None] = 'a7e2c4f86d31'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


GRAPH = "earthlink_social"
VLABELS = ["Agent", "Location"]
ELABELS = ["INTERACTED_WITH", "TAUGHT", "LEARNED_FROM", "BELIEVES_ABOUT", "VISITED"]


def upgrade() -> None:
    op.execute(sa.text("LOAD 'age'"))
    op.execute(sa.text('SET search_path = ag_catalog, "$user", public'))
    op.execute(sa.text(f"SELECT create_graph('{GRAPH}')"))
    for v in VLABELS:
        op.execute(sa.text(f"SELECT create_vlabel('{GRAPH}', '{v}')"))
    for e in ELABELS:
        op.execute(sa.text(f"SELECT create_elabel('{GRAPH}', '{e}')"))


def downgrade() -> None:
    op.execute(sa.text("LOAD 'age'"))
    op.execute(sa.text('SET search_path = ag_catalog, "$user", public'))
    # drop_graph(name, cascade=true) removes the graph schema + all labels/data.
    op.execute(sa.text(f"SELECT drop_graph('{GRAPH}', true)"))
