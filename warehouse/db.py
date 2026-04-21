from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from utils.config import AppConfig


def build_engine(config: AppConfig) -> Engine:
    db_conf = config.database
    return create_engine(
        db_conf["url"],
        echo=bool(db_conf.get("echo", False)),
        pool_pre_ping=bool(db_conf.get("pool_pre_ping", True)),
        pool_recycle=int(db_conf.get("pool_recycle", 3600)),
        future=True,
    )


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

