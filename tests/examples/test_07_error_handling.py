from .test_base import ExampleTest


class TestErrorHandlingExamples(ExampleTest):
    def test_robust_patterns(self) -> None:
        self.run_example_test("07_error_handling/robust_patterns.py")
