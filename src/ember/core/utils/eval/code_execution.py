"""Code execution evaluators for programming problems.

This module provides evaluators for executing and testing code solutions
for programming problems, with appropriate security measures and sandboxing.
"""

import os
import re
import resource
import signal
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

from ember.core.exceptions import DataError, DataTransformationError, ExecutionError
from ember.core.utils.eval.base_evaluator import EvaluationResult, IEvaluator


class CodeExecutionError(ExecutionError):
    """Raised when code execution fails due to runtime errors or security violations."""

    DEFAULT_ERROR_CODE = 4300
    DEFAULT_RECOVERY_HINT = (
        "Check code for syntax errors and ensure it meets security requirements"
    )

    @classmethod
    def for_execution(
        cls,
        language: str,
        error_type: str,
        stderr: Optional[str] = None,
        exit_code: Optional[int] = None,
        **context: Any,
    ) -> "CodeExecutionError":
        """Create an exception for a specific code execution error.

        Args:
            language: Programming language
            error_type: Type of error (e.g., "compilation", "runtime", "timeout")
            stderr: Error output
            exit_code: Process exit code
            **context: Additional context fields

        Returns:
            A new CodeExecutionError
        """
        message = f"Error executing {language} code: {error_type}"

        error_context = {"language": language, "error_type": error_type, **context}

        if stderr:
            # Truncate stderr to avoid excessive context size
            error_context["stderr"] = stderr[:500] + (
                "..." if len(stderr) > 500 else ""
            )

        if exit_code is not None:
            error_context["exit_code"] = exit_code

        return cls(message=message, context=error_context)


class SecurityViolationError(DataTransformationError):
    """Raised when code contains potentially unsafe operations."""

    DEFAULT_ERROR_CODE = 4301
    DEFAULT_RECOVERY_HINT = (
        "Remove unsafe operations like imports, file access, and network calls"
    )

    @classmethod
    def for_pattern(
        cls, pattern: str, code_snippet: Optional[str] = None
    ) -> "SecurityViolationError":
        """Create an exception for a specific security violation pattern.

        Args:
            pattern: The security pattern that was matched
            code_snippet: Optional code snippet context

        Returns:
            A new SecurityViolationError
        """
        message = f"Security violation: detected potentially unsafe pattern '{pattern}'"
        context = {"unsafe_pattern": pattern}

        if code_snippet:
            # Include limited context from the code
            context["code_snippet"] = code_snippet[:200] + (
                "..." if len(code_snippet) > 200 else ""
            )

        return cls(message=message, context=context)


@dataclass
class TestCaseResult:
    """Represents the result of a single test case execution."""

    passed: bool
    execution_time: float
    memory_used_mb: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    exit_code: Optional[int] = None
    security_violations: List[str] = field(default_factory=list)


class LanguageHandler(ABC):
    """Abstract base class for language-specific code execution."""

    @abstractmethod
    def get_file_extension(self) -> str:
        """Return the file extension for this language."""
        pass

    @abstractmethod
    def prepare_code(self, code: str) -> str:
        """Prepare code for execution, performing any necessary transformations."""
        return code

    @abstractmethod
    def get_run_command(self, code_file: Path) -> List[str]:
        """Return the command to run the code."""
        pass

    def get_compile_command(self, code_file: Path) -> Optional[List[str]]:
        """Return the command to compile the code, if needed."""
        return None

    def compare_outputs(
        self, expected: str, actual: str, case_sensitive: bool = True
    ) -> bool:
        """Compare expected and actual outputs, normalizing whitespace."""
        # Normalize line endings
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
        """Prepare Python code for execution with security constraints.

        Applies a series of security measures:
        1. Checks for unsafe imports and operations
        2. Wraps code with safe imports and execution environment
        3. Prevents access to system resources and network

        Args:
            code: The Python code to prepare

        Returns:
            Prepared code with appropriate imports and safety measures

        Raises:
            SecurityViolationError: If code contains unsafe patterns
        """
        # Allowlist of safe imports for competitive programming
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

        # Security patterns to check (pattern, description)
        unsafe_patterns = [
            # System access
            (r"import\s+os", "system module access"),
            (r"import\s+sys", "system module access"),
            (r"import\s+shutil", "filesystem operations"),
            (r"from\s+os\s+import", "system module access"),
            (r"from\s+sys\s+import", "system module access"),
            # Subprocess and shell access
            (r"subprocess\.", "subprocess execution"),
            (r"import\s+subprocess", "subprocess execution"),
            (r"import\s+pty", "terminal access"),
            # Dynamic code execution
            (r"__import__\s*\(", "dynamic import"),
            (r"eval\s*\(", "eval execution"),
            (r"exec\s*\(", "exec execution"),
            (r"compile\s*\(", "code compilation"),
            # File operations
            (r"open\s*\(", "file access"),
            (r"__file__", "file path access"),
            (r"__builtins__", "builtins modification"),
            # Network access
            (r"import\s+socket", "network access"),
            (r"import\s+urllib", "network access"),
            (r"import\s+requests", "network access"),
            (r"import\s+http", "network access"),
            # Other dangerous operations
            (r"globals\s*\(", "globals access"),
            (r"setattr\s*\(", "attribute modification"),
            (r"getattr\s*\(", "attribute access"),
            (r"import\s+ctypes", "C bindings"),
            (r"import\s+multiprocessing", "process spawning"),
        ]

        # Check for unsafe patterns and collect violations
        violations = []
        for pattern, description in unsafe_patterns:
            match = re.search(pattern, code)
            if match:
                snippet = code[
                    max(0, match.start() - 10) : min(len(code), match.end() + 10)
                ]
                violations.append(f"{description} ({pattern}): {snippet}")

        # If any violations, raise a security error
        if violations:
            violation_details = "\n- ".join([f"{v}" for v in violations])
            raise SecurityViolationError.for_pattern(
                pattern="multiple security violations"
                if len(violations) > 1
                else violations[0],
                code_snippet=code[:200] if len(code) > 200 else code,
            )

        # Memory and CPU limits to prevent infinite loops and memory bombs
        resource_limits = [
            "# Set resource limits",
            "import resource",
            "resource.setrlimit(resource.RLIMIT_CPU, (5, 5))  # 5 CPU seconds",
            "resource.setrlimit(resource.RLIMIT_AS, (500 * 1024 * 1024, 500 * 1024 * 1024))  # 500MB memory",
        ]

        # Wrap code with safety measures
        wrapper = "\n".join(
            [
                "# Safe competitive programming environment",
                "\n".join(safe_imports),
                "",
                "# Safety measures",
                "\n".join(resource_limits),
                "",
                "# Begin user code",
                code,
                "# End user code",
            ]
        )

        return wrapper

    def get_run_command(self, code_file: Path) -> List[str]:
        """Return the command to run Python code.

        Args:
            code_file: Path to the Python file.

        Returns:
            List of command parts to execute.
        """
        return ["python", "-u", str(code_file)]  # -u for unbuffered output


class CPPHandler(LanguageHandler):
    """Handler for C++ code execution."""

    def get_file_extension(self) -> str:
        """Return the file extension for C++."""
        return ".cpp"

    def prepare_code(self, code: str) -> str:
        """Prepare C++ code for execution."""
        # In a real implementation, we might add some safety headers
        # or memory limit controls
        return code

    def get_compile_command(self, code_file: Path) -> List[str]:
        """Return the command to compile C++ code.

        Args:
            code_file: Path to the C++ file.

        Returns:
            List of command parts for compilation.
        """
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

    def get_run_command(self, code_file: Path) -> List[str]:
        """Return the command to run compiled C++ code.

        Args:
            code_file: Path to the source C++ file.

        Returns:
            List of command parts to execute the compiled binary.
        """
        executable = code_file.parent / code_file.stem
        return [str(executable)]


class CodeExecutor:
    """Handles secure execution of code with resource limits."""

    def __init__(
        self,
        time_limit: float = 2.0,
        memory_limit_mb: int = 512,
        max_output_size: int = 1024 * 1024,  # 1MB max output size
    ) -> None:
        """Initialize the code executor with resource constraints.

        Args:
            time_limit: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
            max_output_size: Maximum allowed output size in bytes
        """
        self.time_limit = time_limit
        self.memory_limit_mb = memory_limit_mb
        self.max_output_size = max_output_size

        # Register language handlers
        self.handlers = {
            "python": PythonHandler(),
            "cpp": CPPHandler(),
            # Add more language handlers as needed
        }

    def get_handler(self, language: str) -> LanguageHandler:
        """Get the appropriate language handler for a programming language.

        Args:
            language: The programming language identifier

        Returns:
            Language-specific handler

        Raises:
            DataError: If the language is not supported
        """
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

    def _monitor_process_resources(self, process: subprocess.Popen) -> float:
        """Monitor memory usage of a running process.

        Args:
            process: Running subprocess to monitor

        Returns:
            Peak memory usage in MB or 0.0 if monitoring fails
        """
        try:
            import psutil

            # Try to create a psutil process from subprocess pid
            p = psutil.Process(process.pid)
            peak_memory = 0.0

            # Check memory usage while process is running
            while process.poll() is None:
                try:
                    # Get memory info for the process
                    mem_info = p.memory_info()
                    current_memory = mem_info.rss / (1024 * 1024)  # Convert to MB
                    peak_memory = max(peak_memory, current_memory)

                    # Short sleep to avoid excessive CPU usage
                    time.sleep(0.05)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

            return peak_memory
        except (ImportError, PermissionError):
            # If psutil isn't available or we don't have permission, just return 0
            return 0.0

    def run_code(
        self, code: str, language: str, input_data: str, timeout: Optional[float] = None
    ) -> TestCaseResult:
        """Execute code with the given input in a secure environment.

        Runs code within a temporary directory with strict resource limits.
        Handles compilation for compiled languages, execution for all languages,
        and proper cleanup regardless of execution outcome.

        Args:
            code: Source code to execute
            language: Programming language identifier
            input_data: Input data to provide to the program
            timeout: Optional custom timeout in seconds

        Returns:
            TestCaseResult with execution details

        Raises:
            SecurityViolationError: If code contains unsafe patterns
            CodeExecutionError: If execution preparation fails
        """
        if timeout is None:
            timeout = self.time_limit

        # Get the language handler
        handler = self.get_handler(language)

        # Create a temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Prepare code file with security checks
                prepared_code = handler.prepare_code(code)
                extension = handler.get_file_extension()
                code_file = temp_path / f"solution{extension}"

                with open(code_file, "w") as f:
                    f.write(prepared_code)

                # Prepare input file
                input_file = temp_path / "input.txt"
                with open(input_file, "w") as f:
                    f.write(input_data)

                # Compile if necessary (for compiled languages)
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

                        if compile_result.returncode != 0:
                            return TestCaseResult(
                                passed=False,
                                execution_time=0.0,
                                memory_used_mb=0.0,
                                stdout="",
                                stderr=compile_result.stderr,
                                error="Compilation error",
                                exit_code=compile_result.returncode,
                            )
                    except subprocess.TimeoutExpired:
                        return TestCaseResult(
                            passed=False,
                            execution_time=timeout,
                            error="Compilation timeout",
                            exit_code=None,
                        )

                # Execute code with resource monitoring
                run_cmd = handler.get_run_command(code_file)
                start_time = time.time()
                memory_used = 0.0

                try:
                    with open(input_file, "r") as f_in:
                        # Start process with Popen so we can monitor resources
                        process = subprocess.Popen(
                            run_cmd,
                            cwd=temp_dir,
                            stdin=f_in,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            # Set process group to enable clean termination of subprocesses
                            preexec_fn=os.setsid,
                        )

                        # Start resource monitoring in a separate thread
                        import threading

                        monitor_thread = threading.Thread(
                            target=lambda: setattr(
                                threading.current_thread(),
                                "memory_usage",
                                self._monitor_process_resources(process),
                            )
                        )
                        monitor_thread.daemon = True
                        monitor_thread.start()

                        try:
                            # Wait for process with timeout
                            stdout, stderr = process.communicate(timeout=timeout)
                            execution_time = time.time() - start_time

                            # Check if monitor thread has memory usage data
                            monitor_thread.join(0.1)
                            if hasattr(monitor_thread, "memory_usage"):
                                memory_used = getattr(monitor_thread, "memory_usage")

                            # Truncate large outputs
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
                                passed=(
                                    process.returncode == 0
                                ),  # Will be updated by evaluator
                                execution_time=execution_time,
                                memory_used_mb=memory_used,
                                stdout=stdout,
                                stderr=stderr,
                                error=None
                                if process.returncode == 0
                                else "Runtime error",
                                exit_code=process.returncode,
                            )

                        except subprocess.TimeoutExpired:
                            # Clean termination of process group
                            try:
                                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            except (ProcessLookupError, PermissionError):
                                pass

                            # Clean up and return timeout result
                            process.kill()
                            execution_time = time.time() - start_time

                            return TestCaseResult(
                                passed=False,
                                execution_time=execution_time,
                                memory_used_mb=memory_used,
                                error="Time limit exceeded",
                                exit_code=None,
                            )

                except Exception as e:
                    # Handle any unexpected errors during execution
                    execution_time = time.time() - start_time
                    return TestCaseResult(
                        passed=False,
                        execution_time=execution_time,
                        error=f"Execution error: {str(e)}",
                        exit_code=None,
                    )

            except SecurityViolationError:
                # Re-raise security violations
                raise

            except Exception as e:
                # Wrap other exceptions in CodeExecutionError
                raise CodeExecutionError.for_execution(
                    language=language, error_type="preparation_error", stderr=str(e)
                ) from e


class CodeCompetitionEvaluator(IEvaluator[str, Dict[str, Any]]):
    """Evaluator for competitive programming problems.

    Executes submitted code against multiple test cases in a controlled
    environment, with time limits and memory monitoring. Provides detailed
    execution metrics for each test case and aggregated results.
    """

    def __init__(
        self,
        time_limit: float = 2.0,
        memory_limit_mb: int = 512,
        supported_languages: Optional[List[str]] = None,
        max_output_size: int = 1024 * 1024,  # 1MB max output
    ) -> None:
        """Initialize the code competition evaluator with resource limits.

        Args:
            time_limit: Maximum execution time per test case (seconds)
            memory_limit_mb: Maximum memory usage allowed (MB)
            supported_languages: List of supported languages, defaults to ["python"]
            max_output_size: Maximum allowed output size in bytes
        """
        self.time_limit = time_limit
        self.memory_limit_mb = memory_limit_mb
        self.supported_languages = supported_languages or ["python"]
        self.executor = CodeExecutor(
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
            max_output_size=max_output_size,
        )

    def evaluate(
        self, system_output: str, reference_data: Dict[str, Any], **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate generated code against test cases.

        Executes code against each test case, compares outputs, and aggregates results.
        Handles various error conditions gracefully, providing meaningful error messages
        and diagnostic information.

        Args:
            system_output: Generated code solution
            reference_data: Dictionary containing test cases and expected outputs
            **kwargs: Additional parameters including:
                - language: Programming language (default: "python")
                - case_sensitive: Whether output comparison is case-sensitive (default: True)
                - detailed_results: Whether to include full test details (default: True)

        Returns:
            EvaluationResult with test case results and execution metrics
        """
        language = kwargs.get("language", "python").lower()
        case_sensitive = kwargs.get("case_sensitive", True)
        include_details = kwargs.get("detailed_results", True)

        # Check language support
        if language not in self.supported_languages:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={
                    "error": f"Unsupported language: {language}",
                    "error_type": "unsupported_language",
                    "supported_languages": self.supported_languages,
                },
            )

        # Validate test cases
        test_cases = reference_data.get("test_cases", [])
        if not test_cases:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={
                    "error": "No test cases provided",
                    "error_type": "missing_test_cases",
                },
            )

        # Process test cases
        results = []
        passed_count = 0
        total_time = 0.0
        max_memory = 0.0
        error_message = None

        try:
            # Get handler for comparing outputs
            handler = self.executor.get_handler(language)

            for i, test_case in enumerate(test_cases):
                test_id = i + 1
                input_data = test_case.get("input", "")
                expected_output = test_case.get("output", "")

                try:
                    # Execute code on this test case
                    test_result = self.executor.run_code(
                        system_output, language, input_data, self.time_limit
                    )

                    # Compare output if no runtime error
                    if test_result.passed:
                        output_matches = handler.compare_outputs(
                            expected_output, test_result.stdout, case_sensitive
                        )
                        test_result.passed = output_matches

                    # Update statistics
                    if test_result.passed:
                        passed_count += 1

                    total_time += test_result.execution_time
                    max_memory = max(max_memory, test_result.memory_used_mb)

                    # Build result with appropriate detail level
                    result_entry = {
                        "test_case": test_id,
                        "passed": test_result.passed,
                        "execution_time": round(test_result.execution_time, 4),
                        "memory_used_mb": round(test_result.memory_used_mb, 2),
                        "error": test_result.error,
                    }

                    # Add output previews if detailed results requested
                    if include_details:
                        # Truncate long outputs for metadata
                        result_entry.update(
                            {
                                "stderr_preview": (test_result.stderr[:200] + "...")
                                if len(test_result.stderr) > 200
                                else test_result.stderr,
                                "output_preview": (test_result.stdout[:200] + "...")
                                if len(test_result.stdout) > 200
                                else test_result.stdout,
                                "expected_preview": (expected_output[:200] + "...")
                                if len(expected_output) > 200
                                else expected_output,
                            }
                        )

                    results.append(result_entry)

                except SecurityViolationError as e:
                    # Security violations should stop the entire evaluation
                    return EvaluationResult(
                        is_correct=False,
                        score=0.0,
                        metadata={
                            "error": f"Security violation: {e.message}",
                            "error_type": "security_violation",
                            "language": language,
                            "test_case": test_id,
                            "context": e.context,
                        },
                    )

                except CodeExecutionError as e:
                    # Code execution errors are per-test-case
                    results.append(
                        {
                            "test_case": test_id,
                            "passed": False,
                            "execution_time": 0.0,
                            "error": f"Execution error: {e.message}",
                            "error_type": e.context.get(
                                "error_type", "execution_error"
                            ),
                        }
                    )

                except Exception as e:
                    # Unexpected errors during individual test case
                    results.append(
                        {
                            "test_case": test_id,
                            "passed": False,
                            "execution_time": 0.0,
                            "error": f"Unexpected error: {str(e)}",
                            "error_type": "unknown_error",
                        }
                    )

        except Exception as e:
            # Handle any unexpected errors during overall evaluation
            error_message = str(e)
            if isinstance(e, DataError):
                error_type = "data_error"
            elif isinstance(e, ExecutionError):
                error_type = "execution_error"
            else:
                error_type = "unknown_error"

            metadata = {
                "error": f"Evaluation error: {error_message}",
                "error_type": error_type,
                "language": language,
            }

            # Include context from EmberErrors
            if hasattr(e, "context") and isinstance(e.context, dict):
                metadata["context"] = e.context

            return EvaluationResult(is_correct=False, score=0.0, metadata=metadata)

        # Calculate final score and result
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
                "avg_execution_time": round(
                    total_time / total_cases if total_cases > 0 else 0, 4
                ),
                "max_memory_used_mb": round(max_memory, 2),
                "test_results": results,
                "error_message": error_message,
            },
        )
