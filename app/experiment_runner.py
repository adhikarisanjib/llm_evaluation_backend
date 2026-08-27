import asyncio
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.code_parser import extract_code
from app.llm_clients.ollama import OllamaAdapter
from app.llm_clients.openai import OpenAIAdapter
from app.models import Experiment, ExperimentStatus, Run, RunTestResult
from app.prompt_builder import build_prompt
from app.sandbox import PythonSandbox
from app.settings import settings


class ExperimentRunner:
    def __init__(self, db: AsyncSession, sandbox: PythonSandbox | None = None):
        self.db = db
        self.sandbox = sandbox or PythonSandbox()
        self.settings = settings

    async def run(self, experiment_id: int) -> Experiment:
        statement = (
            select(Experiment)
            .options(
                selectinload(Experiment.models),
                selectinload(Experiment.tasks),
            )
            .where(Experiment.id == experiment_id)
        )
        result = await self.db.execute(statement)
        experiment = result.scalar_one()

        # An experiment defines fixed attempt slots. Re-calling /run is a
        # resumable/idempotent operation: if every model x task x attempt slot
        # already exists, return the existing experiment without regenerating
        # responses or changing its timestamps.
        existing_result = await self.db.execute(
            select(Run.model_id, Run.task_id, Run.attempt_number).where(
                Run.experiment_id == experiment.id
            )
        )
        existing_keys = set(existing_result.all())
        expected_keys = {
            (model.id, task.id, attempt)
            for model in experiment.models
            for task in experiment.tasks
            for attempt in range(1, experiment.attempts_per_task + 1)
        }
        if expected_keys and expected_keys.issubset(existing_keys):
            return experiment

        experiment.status = ExperimentStatus.RUNNING.value
        if experiment.started_at is None:
            experiment.started_at = datetime.now(UTC)
        experiment.completed_at = None
        await self.db.commit()

        try:
            for model in experiment.models:

                for task in experiment.tasks:
                    for attempt in range(1, experiment.attempts_per_task + 1):
                        result = await self.db.execute(
                            select(Run).where(
                                Run.experiment_id == experiment.id,
                                Run.model_id == model.id,
                                Run.task_id == task.id,
                                Run.attempt_number == attempt,
                            )
                        )
                        existing = result.scalar_one_or_none()
                        if existing:
                            continue

                        prompt = build_prompt(task)
                        run = Run(
                            experiment_id=experiment.id,
                            model_id=model.id,
                            task_id=task.id,
                            attempt_number=attempt,
                            prompt=prompt,
                            status="running",
                        )
                        self.db.add(run)
                        await self.db.commit()
                        await self.db.refresh(run)

                        try:
                            if task.programming_language.lower() != "python":
                                raise ValueError(
                                    f"Execution for programming language "
                                    f"{task.programming_language!r} is not implemented yet. "
                                    "The current sandbox supports Python."
                                )

                            if model.adapter_type == "ollama":
                                adapter = OllamaAdapter()
                            elif model.adapter_type == "openai":
                                adapter = OpenAIAdapter()
                            else:
                                raise ValueError(
                                    f"Unknown adapter {model.adapter_type!r} for model {model.name!r}"
                                )

                            response = await adapter.generate(
                                model_config=model,
                                prompt=prompt,
                                settings={
                                    "temperature": experiment.temperature,
                                    "top_p": experiment.top_p,
                                    "max_tokens": experiment.max_tokens,
                                },
                            )

                            code = extract_code(response.content)
                            run.raw_response = response.content
                            run.extracted_code = code
                            run.input_tokens = response.input_tokens
                            run.output_tokens = response.output_tokens
                            run.latency_ms = response.latency_ms
                            run.tokens_per_second = response.tokens_per_second

                            sandbox_result = await asyncio.to_thread(
                                self.sandbox.evaluate,
                                code,
                                task.tests,
                                task.regression_tests,
                                self.settings.sandbox_timeout_seconds,
                            )

                            for result in sandbox_result.results:
                                self.db.add(
                                    RunTestResult(
                                        run_id=run.id,
                                        test_group=result.test_group,
                                        test_index=result.test_index,
                                        test_code=result.test_code,
                                        passed=result.passed,
                                        duration_ms=result.duration_ms,
                                        error=result.error,
                                    )
                                )

                            run.tests_total = len(sandbox_result.results)
                            run.tests_passed = sum(
                                1 for result in sandbox_result.results if result.passed
                            )
                            run.all_tests_passed = (
                                run.tests_total > 0
                                and run.tests_passed == run.tests_total
                            )

                            if task.task_type == "debugging":
                                bug_results = [
                                    result
                                    for result in sandbox_result.results
                                    if result.test_group == "tests"
                                ]
                                regression_results = [
                                    result
                                    for result in sandbox_result.results
                                    if result.test_group == "regression_tests"
                                ]
                                run.bug_fixed = bool(bug_results) and all(
                                    result.passed for result in bug_results
                                )
                                run.regression_count = sum(
                                    1
                                    for result in regression_results
                                    if not result.passed
                                )

                            run.status = "completed"

                        except Exception as exc:
                            run.status = "failed"
                            run.error = f"{type(exc).__name__}: {exc}"

                        await self.db.commit()

            experiment.status = ExperimentStatus.COMPLETED.value
            experiment.completed_at = datetime.now(UTC)
            await self.db.commit()
            await self.db.refresh(experiment)
            return experiment

        except Exception:
            experiment.status = ExperimentStatus.FAILED.value
            experiment.completed_at = datetime.now(UTC)
            await self.db.commit()
            raise

    async def retry_failed(self, experiment_id: int) -> Experiment:
        """Retry only failed attempt slots while preserving completed runs."""
        result = await self.db.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        )
        experiment = result.scalar_one_or_none()
        if experiment is None:
            raise ValueError("Experiment not found")

        failed_result = await self.db.execute(
            select(Run).where(
                Run.experiment_id == experiment_id,
                Run.status == "failed",
            )
        )
        failed_runs = failed_result.scalars().all()
        for failed_run in failed_runs:
            await self.db.delete(failed_run)
        await self.db.commit()

        return await self.run(experiment_id)
