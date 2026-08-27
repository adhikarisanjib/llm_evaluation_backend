import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.settings import settings


@dataclass
class SingleTestResult:
    test_group: str
    test_index: int
    test_code: str
    passed: bool
    duration_ms: float | None
    error: str | None


@dataclass
class SandboxResult:
    results: list[SingleTestResult]
    stdout: str = ""
    stderr: str = ""


RUNNER_SOURCE = r"""
import contextlib, io, json, runpy, sys, time, traceback
payload = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
solution_path = sys.argv[2]
results = []
buffer_out, buffer_err = io.StringIO(), io.StringIO()
try:
    with contextlib.redirect_stdout(buffer_out), contextlib.redirect_stderr(buffer_err):
        namespace = runpy.run_path(solution_path)
except BaseException as exc:
    error = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
    for test in payload['tests']:
        results.append({
            'test_group': test['test_group'],
            'test_index': test['test_index'],
            'test_code': test['test_code'],
            'passed': False,
            'duration_ms': 0.0,
            'error': 'solution import failed: ' + error,
        })
else:
    for test in payload['tests']:
        started = time.perf_counter()
        passed, error = True, None
        try:
            with contextlib.redirect_stdout(buffer_out), contextlib.redirect_stderr(buffer_err):
                exec(test['test_code'], namespace, namespace)
        except BaseException as exc:
            passed = False
            error = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
        results.append({
            'test_group': test['test_group'],
            'test_index': test['test_index'],
            'test_code': test['test_code'],
            'passed': passed,
            'duration_ms': (time.perf_counter() - started) * 1000,
            'error': error,
        })
print(json.dumps({
    'results': results,
    'captured_stdout': buffer_out.getvalue(),
    'captured_stderr': buffer_err.getvalue(),
}))
"""


class PythonSandbox:
    def __init__(self):
        self.settings = settings

    def evaluate(
        self,
        code: str,
        tests: list[str],
        regression_tests: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> SandboxResult:
        regression_tests = regression_tests or []
        timeout_seconds = timeout_seconds or self.settings.sandbox_timeout_seconds

        test_payload = [
            {
                "test_group": "tests",
                "test_index": index,
                "test_code": test_code,
            }
            for index, test_code in enumerate(tests)
        ] + [
            {
                "test_group": "regression_tests",
                "test_index": index,
                "test_code": test_code,
            }
            for index, test_code in enumerate(regression_tests)
        ]

        with tempfile.TemporaryDirectory(prefix="llm-eval-") as temp:
            root = Path(temp)
            (root / "solution.py").write_text(code, encoding="utf-8")
            (root / "eval_runner.py").write_text(RUNNER_SOURCE, encoding="utf-8")
            (root / "tests.json").write_text(
                json.dumps({"tests": test_payload}),
                encoding="utf-8",
            )

            # TemporaryDirectory is created with mode 0700. The Docker image
            # deliberately runs as the unprivileged `runner` user (UID 10001),
            # so that user cannot traverse a 0700 bind-mounted workspace owned
            # by the host FastAPI process. Make only this disposable workspace
            # traversable/readable before mounting it read-only into Docker.
            root.chmod(0o755)
            (root / "solution.py").chmod(0o444)
            (root / "eval_runner.py").chmod(0o444)
            (root / "tests.json").chmod(0o444)

            if self.settings.sandbox_backend == "local":
                cmd = [
                    "python",
                    str(root / "eval_runner.py"),
                    str(root / "tests.json"),
                    str(root / "solution.py"),
                ]
            else:
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--memory",
                    "512m",
                    "--cpus",
                    "1",
                    "--pids-limit",
                    "64",
                    "--read-only",
                    "-v",
                    f"{root}:/workspace:ro",
                    self.settings.sandbox_image,
                    "python",
                    "/workspace/eval_runner.py",
                    "/workspace/tests.json",
                    "/workspace/solution.py",
                ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Docker is not installed or not on PATH. Build/use the sandbox "
                    "or set SANDBOX_BACKEND=local only for trusted development tests."
                ) from exc
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    [
                        SingleTestResult(
                            item["test_group"],
                            item["test_index"],
                            item["test_code"],
                            False,
                            None,
                            f"execution timed out after {timeout_seconds}s",
                        )
                        for item in test_payload
                    ]
                )

            if proc.returncode != 0:
                message = proc.stderr.strip() or proc.stdout.strip()
                # A non-zero Docker/runner process exit means the evaluation
                # infrastructure itself did not produce test results. Do not
                # turn this into false model test failures because that would
                # contaminate Pass@k and debugging metrics.
                raise RuntimeError(f"sandbox failed: {message}")

            try:
                data = json.loads(proc.stdout.strip().splitlines()[-1])
            except Exception as exc:
                raise RuntimeError(
                    "sandbox returned invalid output: "
                    f"{proc.stdout.strip() or proc.stderr.strip()}"
                ) from exc

            return SandboxResult(
                [SingleTestResult(**item) for item in data["results"]],
                data.get("captured_stdout", ""),
                data.get("captured_stderr", ""),
            )
