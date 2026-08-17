"""SQLAlchemy engine and session factory.

All connection parameters come from ``Backend.config.settings`` (sourced from
``.env``, never hardcoded).  The engine uses the ``psycopg`` (v3) dialect per
D-019.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from Backend.config import settings


engine = create_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def get_session():
    """Yield a transactional session; caller must commit/rollback/close."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
