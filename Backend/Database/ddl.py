"""DDL helpers — create / drop all tables.

Intended for bootstrapping and test setup.  NOT called at application startup
in production; schema migrations would be handled by Alembic or equivalent in
a future phase.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from Backend.Database.connection import engine, Base
from Backend.Database.schema import (
    Patient,
    Observation,
    Prediction,
    Alert,
    AlertSummary,
)


def create_all_tables() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def drop_all_tables() -> None:
    """Drop all tables (test teardown only)."""
    Base.metadata.drop_all(bind=engine)


def table_names() -> list[str]:
    """Return the list of table names defined in metadata."""
    return sorted(Base.metadata.tables.keys())


def verify_schema() -> dict[str, list[str]]:
    """Return {table_name: [column_names]} for all defined tables."""
    result = {}
    for name, table in sorted(Base.metadata.tables.items()):
        result[name] = [col.name for col in table.columns]
    return result
