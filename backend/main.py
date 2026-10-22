from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from database import AgentTask, Base, Note, Run, engine, get_db
from agent import run_agent
from evaluator import CASES, evaluate
from providers import configuration_error, runtime_status


@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title='Agent Check', version='2.0.0', lifespan=lifespan)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    agent: Literal['demo_safe', 'demo_unsafe', 'llm', 'autonomous'] = 'demo_safe'


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    goal: str = Field(min_length=1, max_length=4000)
    max_steps: int = Field(default=6, ge=1, le=8, strict=True)

    @field_validator('goal')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Enter a goal.')
        return value.strip()


def require_runtime():
    error = configuration_error()
    if error:
        raise HTTPException(400, error)


def serialize(run):
    created_at = run.created_at
    if created_at.tzinfo is None:  # SQLite does not retain timezone metadata.
        created_at = created_at.replace(tzinfo=timezone.utc)
    return {'id': run.id, 'created_at': created_at.isoformat(),
            'agent': run.agent, **run.report}


@app.get('/api/health')
def health(db: Session = Depends(get_db)):
    db.execute(text('SELECT 1'))
    return {'status': 'ok', 'database': engine.dialect.name, **runtime_status()}


@app.get('/api/cases')
def cases():
    return CASES


@app.post('/api/runs', status_code=201)
def create_run(request: RunRequest, db: Session = Depends(get_db)):
    if request.agent in ('llm', 'autonomous'):
        require_runtime()
    run = Run(id=str(uuid4()), created_at=datetime.now(timezone.utc),
              agent=request.agent, report=evaluate(request.agent, db=db))
    db.add(run)
    db.commit()
    db.refresh(run)
    return serialize(run)


@app.get('/api/runs')
def list_runs(db: Session = Depends(get_db)):
    return [serialize(run) for run in db.scalars(
        select(Run).order_by(Run.created_at.desc()).limit(50))]


@app.get('/api/runs/{run_id}')
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(404, 'Run not found.')
    return serialize(run)


def serialize_task(task):
    created_at = task.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return {'id': task.id, 'created_at': created_at.isoformat(),
            'goal': task.goal, **task.report}


@app.post('/api/tasks', status_code=201)
def create_task(request: TaskRequest, db: Session = Depends(get_db)):
    require_runtime()
    task = AgentTask(id=str(uuid4()), created_at=datetime.now(timezone.utc), goal=request.goal,
                     report={'status': 'running', 'answer': '', 'trace': []})
    db.add(task)
    db.commit()
    task.report = run_agent(request.goal, db, max_steps=request.max_steps)
    db.commit()
    db.refresh(task)
    return serialize_task(task)


@app.get('/api/tasks')
def list_tasks(db: Session = Depends(get_db)):
    return [serialize_task(task) for task in db.scalars(
        select(AgentTask).order_by(AgentTask.created_at.desc()).limit(30))]


@app.get('/api/tasks/{task_id}')
def get_task(task_id: str, db: Session = Depends(get_db)):
    task = db.get(AgentTask, task_id)
    if not task:
        raise HTTPException(404, 'Task not found.')
    return serialize_task(task)


@app.get('/api/notes')
def list_notes(db: Session = Depends(get_db)):
    return [{'title': note.title, 'content': note.content}
            for note in db.scalars(select(Note).order_by(Note.title).limit(100))]
