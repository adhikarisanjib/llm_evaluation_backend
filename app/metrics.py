from collections import defaultdict
from math import comb

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Experiment, Run

DIFFICULTIES = ("easy", "medium", "hard")


def _average(values):
    """
    Return the arithmetic mean of non-None values.
    """

    values = [value for value in values if value is not None]

    if not values:
        return None

    return sum(values) / len(values)


def _pass_at_k(n: int, c: int, k: int) -> float | None:
    """
    Standard Pass@k estimator.

    n = number of independent attempts for one task
    c = number of completely successful attempts
    k = requested k value

    Example:
        n = 10 attempts
        c = 4 successful attempts
        k = 5
    """

    if n < k:
        return None

    if c == 0:
        return 0.0

    if n - c < k:
        return 1.0

    return 1.0 - (comb(n - c, k) / comb(n, k))


def _calculate_pass_at_k(
    runs: list[Run],
    k: int,
) -> float | None:
    """
    Calculate Pass@k across the tasks represented by `runs`.

    Runs are first grouped by task.

    Pass@k is calculated for every task and then averaged
    across tasks.
    """

    if not runs:
        return None

    runs_by_task = defaultdict(list)

    for run in runs:
        runs_by_task[run.task_id].append(run)

    task_values = []

    for task_runs in runs_by_task.values():

        n = len(task_runs)

        c = sum(1 for run in task_runs if run.all_tests_passed is True)

        value = _pass_at_k(n=n, c=c, k=k)

        if value is not None:
            task_values.append(value)

    return _average(task_values)


def _test_metrics(
    runs: list[Run],
):
    """
    Calculate normal test metrics.

    Regression tests are NOT included here.
    """

    results = [
        test for run in runs for test in run.test_results if test.test_group == "tests"
    ]

    total = len(results)

    passed = sum(1 for test in results if test.passed is True)

    failed = total - passed

    pass_rate = passed / total if total > 0 else None

    return {
        "total_tests": total,
        "tests_passed": passed,
        "tests_failed": failed,
        "test_case_pass_rate": pass_rate,
    }


def _bug_fix_metrics(
    runs: list[Run],
):
    """
    Calculate bug-fix success rate.

    This should only receive debugging runs.
    """

    total = len(runs)

    successes = sum(1 for run in runs if run.bug_fixed is True)

    failures = total - successes

    rate = successes / total if total > 0 else None

    return {
        "bug_fix_successes": successes,
        "bug_fix_failures": failures,
        "bug_fix_success_rate": rate,
    }


def _regression_metrics(runs: list[Run]):
    """
    Calculate regression metrics.

    Only regression_tests are considered.
    """

    results = [
        test
        for run in runs
        for test in run.test_results
        if test.test_group == "regression_tests"
    ]

    total = len(results)

    passed = sum(1 for test in results if test.passed is True)

    failed = total - passed

    regression_rate = failed / total if total > 0 else None

    regression_test_pass_rate = passed / total if total > 0 else None

    return {
        "regression_tests_total": total,
        "regression_tests_passed": passed,
        "regression_tests_failed": failed,
        "regression_rate": regression_rate,
        "regression_test_pass_rate": (regression_test_pass_rate),
    }


def _performance_metrics(
    runs: list[Run],
):
    """
    Calculate average performance metrics.
    """

    average_latency_ms = _average([run.latency_ms for run in runs])

    average_tokens_per_second = _average([run.tokens_per_second for run in runs])

    return {
        "average_latency_ms": average_latency_ms,
        "average_tokens_per_second": (average_tokens_per_second),
    }


def _task_group_metrics(
    runs: list[Run],
    debugging: bool = False,
):
    """
    Metrics used inside:

        code_generation
            easy
            medium
            hard

    or:

        debugging
            easy
            medium
            hard
    """

    tests = _test_metrics(runs)

    performance = _performance_metrics(runs)

    data = {
        "runs": len(runs),
        "pass_at_1": (_calculate_pass_at_k(runs, 1)),
        "pass_at_5": (_calculate_pass_at_k(runs, 5)),
        "test_case_pass_rate": (tests["test_case_pass_rate"]),
        "average_latency_ms": (performance["average_latency_ms"]),
        "average_tokens_per_second": (performance["average_tokens_per_second"]),
    }

    if debugging:

        bug_metrics = _bug_fix_metrics(runs)

        data["bug_fix_success_rate"] = bug_metrics["bug_fix_success_rate"]

    return data


def _task_type_metrics(
    runs: list[Run],
    debugging: bool = False,
):
    """
    Create:

    {
        runs,
        pass_at_1,
        pass_at_5,
        test_case_pass_rate,
        average_latency_ms,
        ...

        easy: {...},
        medium: {...},
        hard: {...}
    }
    """

    data = _task_group_metrics(runs, debugging=debugging)

    for difficulty in DIFFICULTIES:

        difficulty_runs = [
            run
            for run in runs
            if (run.task.difficulty and run.task.difficulty.lower() == difficulty)
        ]

        data[difficulty] = _task_group_metrics(difficulty_runs, debugging=debugging)

    return data


async def experiment_metrics(db: AsyncSession, experiment_id: int):
    """
    Return metrics grouped by model.

    Structure:

    {
        "1": {
            model_id,
            model_name,

            overall metrics...,

            code_generation: {
                overall metrics,
                easy: {...},
                medium: {...},
                hard: {...}
            },

            debugging: {
                overall metrics,
                easy: {...},
                medium: {...},
                hard: {...}
            }
        }
    }
    """

    experiment = await db.get(Experiment, experiment_id)

    if not experiment:
        return {}

    result = await db.execute(
        select(Run)
        .options(
            selectinload(Run.model),
            selectinload(Run.task),
            selectinload(Run.test_results),
        )
        .where(Run.experiment_id == experiment_id, Run.status == "completed")
        .order_by(Run.model_id, Run.task_id, Run.attempt_number)
    )

    runs = result.scalars().all()

    runs_by_model = defaultdict(list)

    for run in runs:
        runs_by_model[run.model_id].append(run)

    output = {}

    for (
        model_id,
        model_runs,
    ) in runs_by_model.items():

        model = model_runs[0].model

        code_generation_runs = [
            run for run in model_runs if (run.task.task_type == "code_generation")
        ]

        debugging_runs = [
            run for run in model_runs if (run.task.task_type == "debugging")
        ]

        test_metrics = _test_metrics(model_runs)

        bug_metrics = _bug_fix_metrics(debugging_runs)

        regression_metrics = _regression_metrics(debugging_runs)

        performance = _performance_metrics(model_runs)

        output[str(model_id)] = {
            # MODEL
            "model_id": model.id,
            "model_name": (model.display_name or model.name),
            # RUN COUNTS
            "runs": len(model_runs),
            "code_generation_runs": len(code_generation_runs),
            "debugging_runs": len(debugging_runs),
            # OVERALL PASS@K
            #
            # Includes BOTH code generation and debugging.
            "pass_at_1": (_calculate_pass_at_k(model_runs, 1)),
            "pass_at_5": (_calculate_pass_at_k(model_runs, 5)),
            # NORMAL TESTS
            "total_tests": (test_metrics["total_tests"]),
            "tests_passed": (test_metrics["tests_passed"]),
            "tests_failed": (test_metrics["tests_failed"]),
            "test_case_pass_rate": (test_metrics["test_case_pass_rate"]),
            # BUG FIX
            "bug_fix_successes": (bug_metrics["bug_fix_successes"]),
            "bug_fix_failures": (bug_metrics["bug_fix_failures"]),
            "bug_fix_success_rate": (bug_metrics["bug_fix_success_rate"]),
            # REGRESSION
            "regression_tests_total": (regression_metrics["regression_tests_total"]),
            "regression_tests_passed": (regression_metrics["regression_tests_passed"]),
            "regression_tests_failed": (regression_metrics["regression_tests_failed"]),
            "regression_rate": (regression_metrics["regression_rate"]),
            "regression_test_pass_rate": (
                regression_metrics["regression_test_pass_rate"]
            ),
            # PERFORMANCE
            "average_latency_ms": (performance["average_latency_ms"]),
            "average_tokens_per_second": (performance["average_tokens_per_second"]),
            # CODE GENERATION BREAKDOWN
            "code_generation": (
                _task_type_metrics(code_generation_runs, debugging=False)
            ),
            # DEBUGGING BREAKDOWN
            "debugging": (_task_type_metrics(debugging_runs, debugging=True)),
        }

    return output
