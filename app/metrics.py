from collections import defaultdict
from math import comb

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Experiment, Run


def _pass_at_k(n: int, c: int, k: int) -> float | None:
    """
    Standard unbiased Pass@k estimator.

    n = total independent generations for a task
    c = correct generations for that task
    k = Pass@k value
    """
    if n < k:
        return None

    if n - c < k:
        return 1.0

    return 1.0 - (comb(n - c, k) / comb(n, k))


async def experiment_metrics(
    db: AsyncSession,
    experiment_id: int,
):
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    result = await db.execute(
        select(Run)
        .options(
            selectinload(Run.model),
            selectinload(Run.task),
            selectinload(Run.test_results),
        )
        .where(
            Run.experiment_id == experiment_id,
            Run.status == "completed",
        )
    )
    runs = result.scalars().all()

    # Group completed runs by model
    runs_by_model = defaultdict(list)

    for run in runs:
        runs_by_model[run.model_id].append(run)

    metrics = {}

    for model_id, model_runs in runs_by_model.items():

        model = model_runs[0].model

        total_runs = len(model_runs)

        normal_test_results = [
            result
            for run in model_runs
            for result in run.test_results
            if result.test_group == "tests"
        ]

        tests_total = len(normal_test_results)

        tests_passed = sum(1 for result in normal_test_results if result.passed)

        tests_failed = tests_total - tests_passed

        test_case_pass_rate = tests_passed / tests_total if tests_total > 0 else None

        debugging_runs = [
            run for run in model_runs if run.task.task_type == "debugging"
        ]

        debugging_runs_total = len(debugging_runs)

        bug_fix_successes = sum(1 for run in debugging_runs if run.bug_fixed is True)

        bug_fix_failures = debugging_runs_total - bug_fix_successes

        bug_fix_success_rate = (
            bug_fix_successes / debugging_runs_total
            if debugging_runs_total > 0
            else None
        )

        regression_results = [
            result
            for run in debugging_runs
            for result in run.test_results
            if result.test_group == "regression_tests"
        ]

        regression_tests_total = len(regression_results)

        regression_tests_passed = sum(
            1 for result in regression_results if result.passed
        )

        regression_tests_failed = regression_tests_total - regression_tests_passed

        regression_rate = (
            regression_tests_failed / regression_tests_total
            if regression_tests_total > 0
            else None
        )

        regression_test_pass_rate = (
            regression_tests_passed / regression_tests_total
            if regression_tests_total > 0
            else None
        )

        generation_runs = [
            run for run in model_runs if run.task.task_type == "code_generation"
        ]

        runs_by_task = defaultdict(list)

        for run in generation_runs:
            runs_by_task[run.task_id].append(run)

        pass_at_1_values = []
        pass_at_5_values = []

        for task_id, task_runs in runs_by_task.items():

            n = len(task_runs)

            # A solution counts as correct only when
            # ALL required tests passed.
            c = sum(1 for run in task_runs if run.all_tests_passed is True)

            p1 = _pass_at_k(
                n=n,
                c=c,
                k=1,
            )

            if p1 is not None:
                pass_at_1_values.append(p1)

            p5 = _pass_at_k(
                n=n,
                c=c,
                k=5,
            )

            if p5 is not None:
                pass_at_5_values.append(p5)

        pass_at_1 = (
            sum(pass_at_1_values) / len(pass_at_1_values) if pass_at_1_values else None
        )

        pass_at_5 = (
            sum(pass_at_5_values) / len(pass_at_5_values) if pass_at_5_values else None
        )

        latency_values = [
            run.latency_ms for run in model_runs if run.latency_ms is not None
        ]

        average_latency_ms = (
            sum(latency_values) / len(latency_values) if latency_values else None
        )

        tokens_per_second_values = [
            run.tokens_per_second
            for run in model_runs
            if run.tokens_per_second is not None
        ]

        average_tokens_per_second = (
            sum(tokens_per_second_values) / len(tokens_per_second_values)
            if tokens_per_second_values
            else None
        )

        metrics[str(model_id)] = {
            # Model
            "model_id": model.id,
            "model_name": (model.display_name or model.name),
            # Run information
            "runs": total_runs,
            # Pass@k
            "pass_at_1": pass_at_1,
            "pass_at_5": pass_at_5,
            # Useful for understanding why Pass@5 may be null
            "code_generation_tasks": len(runs_by_task),
            # Normal tests
            "test_case_pass_rate": test_case_pass_rate,
            "tests_total": tests_total,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            # Debugging
            "bug_fix_success_rate": bug_fix_success_rate,
            "debugging_runs": debugging_runs_total,
            "bug_fix_successes": bug_fix_successes,
            "bug_fix_failures": bug_fix_failures,
            # Regression
            "regression_rate": regression_rate,
            "regression_test_pass_rate": (regression_test_pass_rate),
            "regression_tests_total": (regression_tests_total),
            "regression_tests_passed": (regression_tests_passed),
            "regression_tests_failed": (regression_tests_failed),
            # Performance
            "average_latency_ms": average_latency_ms,
            "average_tokens_per_second": (average_tokens_per_second),
        }

    return metrics
