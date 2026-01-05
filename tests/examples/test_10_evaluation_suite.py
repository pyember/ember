from .test_base import ExampleTest


class TestEvaluationSuiteExamples(ExampleTest):
    def test_accuracy_evaluation(self) -> None:
        self.run_example_test("10_evaluation_suite/accuracy_evaluation.py")

    def test_benchmark_harness(self) -> None:
        self.run_example_test("10_evaluation_suite/benchmark_harness.py")

    def test_consistency_testing(self) -> None:
        self.run_example_test("10_evaluation_suite/consistency_testing.py")
