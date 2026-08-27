from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class TaskType(str, Enum):
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


experiment_models = Table(
    "experiment_models",
    Base.metadata,
    Column(
        "experiment_id",
        ForeignKey("experiments.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "model_id", ForeignKey("llm_models.id", ondelete="CASCADE"), primary_key=True
    ),
)

experiment_tasks = Table(
    "experiment_tasks",
    Base.metadata,
    Column(
        "experiment_id",
        ForeignKey("experiments.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("task_id", ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True),
)


class LLMModel(Base):
    __tablename__ = "llm_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    display_name: Mapped[str] = mapped_column(String(200))
    adapter_type: Mapped[str] = mapped_column(String(50), index=True)
    base_url: Mapped[str] = mapped_column(String(500))
    api_key_env: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    experiments: Mapped[list[Experiment]] = relationship(
        secondary=experiment_models,
        back_populates="models",
    )
    runs: Mapped[list[Run]] = relationship(back_populates="model")

    __table_args__ = (
        UniqueConstraint(
            "adapter_type", "base_url", "name", name="uq_model_endpoint_name"
        ),
    )


class Task(Base):
    """A complete coding-evaluation task.

    Tests live directly on the task as JSON arrays. For a debugging task:
    - tests: tests that prove the bug has been fixed
    - regression_tests: tests for behaviour that must continue to work

    For code generation, regression_tests is normally empty.
    """

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    question: Mapped[str] = mapped_column(Text)
    starter_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    difficulty: Mapped[str | None] = mapped_column(String(30), nullable=True)
    task_type: Mapped[str] = mapped_column(String(30), index=True)
    tests: Mapped[list[str]] = mapped_column(JSON, default=list)
    regression_tests: Mapped[list[str]] = mapped_column(JSON, default=list)
    programming_language: Mapped[str] = mapped_column(
        String(30), default="python", index=True
    )

    experiments: Mapped[list[Experiment]] = relationship(
        secondary=experiment_tasks,
        back_populates="tasks",
    )
    runs: Mapped[list[Run]] = relationship(back_populates="task")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(
        String(30), default=ExperimentStatus.DRAFT.value, index=True
    )
    attempts_per_task: Mapped[int] = mapped_column(Integer, default=1)
    temperature: Mapped[float] = mapped_column(Float, default=0.2)
    top_p: Mapped[float] = mapped_column(Float, default=0.95)
    max_tokens: Mapped[int] = mapped_column(Integer, default=1024)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    models: Mapped[list[LLMModel]] = relationship(
        secondary=experiment_models,
        back_populates="experiments",
    )
    tasks: Mapped[list[Task]] = relationship(
        secondary=experiment_tasks,
        back_populates="experiments",
    )
    runs: Mapped[list[Run]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    model_id: Mapped[int] = mapped_column(ForeignKey("llm_models.id"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)

    prompt: Mapped[str] = mapped_column(Text)
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)

    tests_total: Mapped[int] = mapped_column(Integer, default=0)
    tests_passed: Mapped[int] = mapped_column(Integer, default=0)
    all_tests_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    bug_fixed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    regression_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    model: Mapped[LLMModel] = relationship(back_populates="runs")
    task: Mapped[Task] = relationship(back_populates="runs")
    test_results: Mapped[list[RunTestResult]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RunTestResult.id",
    )

    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "model_id",
            "task_id",
            "attempt_number",
            name="uq_run_attempt",
        ),
    )


class RunTestResult(Base):
    """Result of one test execution for one Run.

    The task owns the test definitions. This table only records what happened in
    a particular run and snapshots the test code for reproducibility.
    """

    __tablename__ = "run_test_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    test_group: Mapped[str] = mapped_column(String(30))  # tests | regression_tests
    test_index: Mapped[int] = mapped_column(Integer)
    test_code: Mapped[str] = mapped_column(Text)
    passed: Mapped[bool] = mapped_column(Boolean)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[Run] = relationship(back_populates="test_results")

    __table_args__ = (
        UniqueConstraint(
            "run_id", "test_group", "test_index", name="uq_run_test_position"
        ),
    )
