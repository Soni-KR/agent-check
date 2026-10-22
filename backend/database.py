"""Evaluation history, autonomous tasks, and notes. SQLite or PostgreSQL."""
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import JSON, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{ROOT / "runs.db"}')
engine = create_engine(
    DATABASE_URL,
    connect_args={'check_same_thread': False} if DATABASE_URL.startswith('sqlite') else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = 'runs'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    agent: Mapped[str] = mapped_column(String(100))
    report: Mapped[dict] = mapped_column(JSON)


class AgentTask(Base):
    __tablename__ = 'agent_tasks'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    goal: Mapped[str] = mapped_column(String(4000))
    report: Mapped[dict] = mapped_column(JSON)


class Note(Base):
    __tablename__ = 'agent_notes'
    title: Mapped[str] = mapped_column(String(80), primary_key=True)
    content: Mapped[str] = mapped_column(String(4000))


def get_db():
    with SessionLocal() as session:
        yield session
