"""Execute candidate solutions against test cases."""

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ember._internal.exceptions import (
    DataError,
    DataTransformationError,
)
from ember.utils.eval.base_evaluator import EvaluationResult, IEvaluator


class SecurityViolationError(DataTransformationError):
    """Raised when code contains potentially unsafe operations."""

    DEFAULT_ERROR_CODE = 4301
    DEFAULT_RECOVERY_HINT = "Remove unsafe operations like imports, file access, and network calls"

    @classmethod
    def for_pattern(
        cls, pattern: str, code_snippet: str | None = None
    ) -> "SecurityViolationError":
        message = f"Security violation: detected potentially unsafe pattern '{pattern}'"
        context = {"unsafe_pattern": pattern}

        if code_snippet:
            context["code_snippet"] = code_snippet[:200] + (
                "..." if len(code_snippet) > 200 else ""
            )

        return cls(message=message, context=context)


@dataclass
class TestCaseResult:
    """Represents the result of a single test case execution."""

    passed: bool
    execution_time: float
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    exit_code: int | None = None


class LanguageHandler(ABC):
    """Abstract base class for language-specific code execution."""

    @abstractmethod
    def get_file_extension(self) -> str:
        raise NotImplementedError

    def prepare_code(self, code: str) -> str:
        return code

    @abstractmethod
    def get_run_command(self, code_file: Path) -> list[str]:
        raise NotImplementedError

    def get_compile_command(self, code_file: Path) -> list[str] | None:
        """Return the command to compile the code, if needed."""
        return None

    def compare_outputs(self, expected: str, actual: str, case_sensitive: bool = True) -> bool:
        expected = expected.replace("\r\n", "\n").strip()
        actual = actual.replace("\r\n", "\n").strip()

        if not case_sensitive:
            expected = expected.lower()
            actual = actual.lower()

        return expected == actual


class PythonHandler(LanguageHandler):
    """Handler for Python code execution."""

    def get_file_extension(self) -> str:
        """Return the file extension for Python."""
        return ".py"

    def prepare_code(self, code: str) -> str:
        """Validate Python code and prepend a minimal allowlist of imports."""
        safe_imports = [
            "import math",
            "import re",
            "import collections",
            "import itertools",
            "import functools",
            "import heapq",
            "import bisect",
            "import random",
            "import string",
            "from collections import Counter, defaultdict, deque",
            "from itertools import combinations, permutations, product",
        ]

        unsafe_patterns = [
            (r"import\s+os", "system module access"),
            (r"import\s+sys", "system module access"),
            (r"import\s+shutil", "filesystem operations"),
            (r"from\s+os\s+import", "system module access"),
            (r"from\s+sys\s+import", "system module access"),
            (r"subprocess\.", "subprocess execution"),
            (r"import\s+subprocess", "subprocess execution"),
            (r"import\s+pty", "terminal access"),
            (r"__import__\s*\(", "dynamic import"),
            (r"eval\s*\(", "eval execution"),
            (r"exec\s*\(", "exec execution"),
            (r"compile\s*\(", "code compilation"),
            (r"open\s*\(", "file access"),
            (r"__file__", "file path access"),
            (r"__builtins__", "builtins modification"),
            (r"import\s+socket", "network access"),
            (r"import\s+urllib", "network access"),
            (r"import\s+requests", "network access"),
            (r"import\s+http", "network access"),
            (r"globals\s*\(", "globals access"),
            (r"setattr\s*\(", "attribute modification"),
            (r"getattr\s*\(", "attribute access"),
            (r"import\s+ctypes", "C bindings"),
            (r"import\s+multiprocessing", "process spawning"),
        ]

        violations = []
        for pattern, description in unsafe_patterns:
            match = re.search(pattern, code)
            if match:
                snippet = code[max(0, match.start() - 10) : min(len(code), match.end() + 10)]
                violations.append(f"{description} ({pattern}): {snippet}")

        if violations:
            raise SecurityViolationError.for_pattern(
                pattern=("multiple security violations" if len(violations) > 1 else violations[0]),
                code_snippet=code[:200] if len(code) > 200 else code,
            )

        return "\n".join([*safe_imports, "", code])

    def get_run_command(self, code_file: Path) -> list[str]:
        return [sys.executable, "-u", str(code_file)]


class CPPHandler(LanguageHandler):
    """Handler for C++ code execution."""

    def get_file_extension(self) -> str:
        return ".cpp"

    def prepare_code(self, code: str) -> str:
        return code

    def get_compile_command(self, code_file: Path) -> list[str]:
        output_file = code_file.parent / code_file.stem
        return [
            "g++",
            "-std=c++17",
            "-O2",
            "-Wall",
            str(code_file),
            "-o",
            str(output_file),
        ]

    def get_run_command(self, code_file: Path) -> list[str]:
        executable = code_file.parent / code_file.stem
        return [str(executable)]


class CodeExecutor:
    """Handles secure execution of code with resource limits."""

    def __init__(
        self,
        time_limit: float = 2.0,
        memory_limit_mb: int = 512,
        max_output_size: int = 1024 * 1024,
    ) -> None:
        if time_limit <= 0:
            raise ValueError(f"time_limit must be > 0, got {time_limit!r}")
        if memory_limit_mb <= 0:
            raise ValueError(f"memory_limit_mb must be > 0, got {memory_limit_mb!r}")
        if max_output_size <= 0:
            raise ValueError(f"max_output_size must be > 0, got {max_output_size!r}")

        self.time_limit = time_limit
        self.memory_limit_mb = memory_limit_mb
        self.max_output_size = max_output_size

        self.handlers = {
            "python": PythonHandler(),
            "cpp": CPPHandler(),
        }

    def get_handler(self, language: str) -> LanguageHandler:
        language = language.lower()
        handler = self.handlers.get(language)
        if handler is None:
            supported = ", ".join(sorted(self.handlers.keys()))
            raise DataError(
                message=f"Unsupported language: {language}",
                context={"language": language, "supported_languages": supported},
                recovery_hint=f"Use one of the supported languages: {supported}",
            )
        return handler

    def run_code(
        self,
        code: str,
        language: str,
        input_data: str,
        timeout: float | None = None,
    ) -> TestCaseResult:
        if timeout is None:
            timeout = self.time_limit

        try:
            import resource
        except ImportError as exc:
            raise DataError(
                message="Code execution sandbox requires 'resource' module support",
                context={"language": language},
                recovery_hint="Run on a Unix-like platform that provides the 'resource' module.",
            ) from exc

        handler = self.get_handler(language)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            prepared_code = handler.prepare_code(code)
            extension = handler.get_file_extension()
            code_file = temp_path / f"solution{extension}"

            with open(code_file, "w") as f:
                f.write(prepared_code)

            input_file = temp_path / "input.txt"
            with open(input_file, "w") as f:
                f.write(input_data)

            compile_cmd = handler.get_compile_command(code_file)
            if compile_cmd:
                try:
                    compile_result = subprocess.run(
                        compile_cmd,
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    return TestCaseResult(
                        passed=False,
                        execution_time=timeout,
                        error="Compilation timeout",
                        exit_code=None,
                    )

                if compile_result.returncode != 0:
                    return TestCaseResult(
                        passed=False,
                        execution_time=0.0,
                        stdout="",
                        stderr=compile_result.stderr,
                        error="Compilation error",
                        exit_code=compile_result.returncode,
                    )

            run_cmd = handler.get_run_command(code_file)
            start_time = time.time()
            with open(input_file, "r") as f_in:
                cpu_limit_seconds = max(1, int(timeout) + 1)
                memory_limit_bytes = self.memory_limit_mb * 1024 * 1024

                def _preexec() -> None:
                    os.setsid()
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds))
                    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))

                process = subprocess.Popen(
                    run_cmd,
                    cwd=temp_dir,
                    stdin=f_in,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=_preexec,
                )

                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass

                    process.kill()
                    execution_time = time.time() - start_time
                    return TestCaseResult(
                        passed=False,
                        execution_time=execution_time,
                        error="Time limit exceeded",
                        exit_code=None,
                    )

            execution_time = time.time() - start_time

            if len(stdout) > self.max_output_size:
                stdout = (
                    stdout[: self.max_output_size]
                    + "\n[Output truncated - exceeded size limit]"
                )
            if len(stderr) > self.max_output_size:
                stderr = (
                    stderr[: self.max_output_size]
                    + "\n[Error output truncated - exceeded size limit]"
                )

            return TestCaseResult(
                passed=process.returncode == 0,
                execution_time=execution_time,
                stdout=stdout,
                stderr=stderr,
                error=(None if process.returncode == 0 else "Runtime error"),
                exit_code=process.returncode,
            )


class CodeCompetitionEvaluator(IEvaluator[str, Mapping[str, object]]):
    """Evaluate code submissions against test cases."""

    def __init__(
        self,
        time_limit: float = 2.0,
        memory_limit_mb: int = 512,
        supported_languages: Sequence[str] | None = None,
        max_output_size: int = 1024 * 1024,
    ) -> None:
        self.time_limit = time_limit
        self.memory_limit_mb = memory_limit_mb
        if supported_languages is None:
            self.supported_languages = ["python"]
        else:
            if not supported_languages:
                raise ValueError("supported_languages must be non-empty when provided")
            self.supported_languages = [lang.strip().lower() for lang in supported_languages]
            if not all(self.supported_languages):
                raise ValueError("supported_languages entries must be non-empty strings")
        self.executor = CodeExecutor(
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
            max_output_size=max_output_size,
        )

    def evaluate(
        self,
        system_output: str,
        reference_data: Mapping[str, object],
        **kwargs: object,
    ) -> EvaluationResult:
        language_value = kwargs.get("language", "python")
        if not isinstance(language_value, str) or not language_value.strip():
            raise TypeError(f"language must be a non-empty string, got {language_value!r}")
        language = language_value.strip().lower()

        case_sensitive_value = kwargs.get("case_sensitive", True)
        if not isinstance(case_sensitive_value, bool):
            raise TypeError(f"case_sensitive must be a bool, got {case_sensitive_value!r}")
        case_sensitive = case_sensitive_value

        include_details_value = kwargs.get("detailed_results", True)
        if not isinstance(include_details_value, bool):
            raise TypeError(f"detailed_results must be a bool, got {include_details_value!r}")
        include_details = include_details_value

        if language not in self.supported_languages:
            supported = ", ".join(self.supported_languages)
            raise DataError(
                message=f"Unsupported language: {language}",
                context={"language": language, "supported_languages": supported},
                recovery_hint=f"Use one of: {supported}",
            )

        test_cases_obj = reference_data.get("test_cases")
        if not isinstance(test_cases_obj, Sequence) or isinstance(test_cases_obj, (str, bytes)):
            raise DataError(
                message="reference_data['test_cases'] must be a sequence of mappings",
                context={"test_cases_type": type(test_cases_obj).__name__},
                recovery_hint=(
                    "Provide reference_data={'test_cases': [{'input': '...', 'output': '...'}]}"
                ),
            )
        if not test_cases_obj:
            raise DataError(
                message="No test cases provided",
                context={},
                recovery_hint="Provide at least one test case in reference_data['test_cases']",
            )
        test_cases: Sequence[object] = test_cases_obj

        results: list[dict[str, object]] = []
        passed_count = 0
        total_time = 0.0
        handler = self.executor.get_handler(language)

        for i, test_case_obj in enumerate(test_cases):
            test_id = i + 1
            if not isinstance(test_case_obj, Mapping):
                raise DataError(
                    message="Each test case must be a mapping",
                    context={
                        "test_case_index": test_id,
                        "test_case_type": type(test_case_obj).__name__,
                    },
                    recovery_hint=(
                        "Provide test cases as {'input': '...', 'output': '...'} mappings."
                    ),
                )

            input_data_obj = test_case_obj.get("input")
            expected_output_obj = test_case_obj.get("output")
            if not isinstance(input_data_obj, str):
                raise DataError(
                    message="Test case input must be a string",
                    context={
                        "test_case_index": test_id,
                        "input_type": type(input_data_obj).__name__,
                    },
                    recovery_hint=(
                        "Provide test cases with {'input': '<stdin>', 'output': '<stdout>'}."
                    ),
                )
            if not isinstance(expected_output_obj, str):
                raise DataError(
                    message="Test case output must be a string",
                    context={
                        "test_case_index": test_id,
                        "output_type": type(expected_output_obj).__name__,
                    },
                    recovery_hint=(
                        "Provide test cases with {'input': '<stdin>', 'output': '<stdout>'}."
                    ),
                )

            test_result = self.executor.run_code(
                system_output, language, input_data_obj, self.time_limit
            )

            if test_result.passed:
                output_matches = handler.compare_outputs(
                    expected_output_obj,
                    test_result.stdout,
                    case_sensitive,
                )
                test_result.passed = output_matches

            if test_result.passed:
                passed_count += 1

            total_time += test_result.execution_time

            result_entry: dict[str, object] = {
                "test_case": test_id,
                "passed": test_result.passed,
                "execution_time": round(test_result.execution_time, 4),
                "error": test_result.error,
            }

            if include_details:
                result_entry.update(
                    {
                        "stderr_preview": test_result.stderr[:200],
                        "output_preview": test_result.stdout[:200],
                        "expected_preview": expected_output_obj[:200],
                    }
                )

            results.append(result_entry)

        total_cases = len(test_cases)
        score = passed_count / total_cases if total_cases > 0 else 0.0
        is_correct = passed_count == total_cases

        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            metadata={
                "passed_count": passed_count,
                "total_cases": total_cases,
                "language": language,
                "total_execution_time": round(total_time, 4),
                "avg_execution_time": round(total_time / total_cases if total_cases > 0 else 0, 4),
                "test_results": results,
            },
        )
