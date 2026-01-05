from .test_base import ExampleTest


class TestPerformanceOptimizationExamples(ExampleTest):
    def test_batch_processing(self) -> None:
        self.run_example_test("06_performance_optimization/batch_processing.py")

    def test_jit_basics(self) -> None:
        self.run_example_test("06_performance_optimization/jit_basics.py")

    def test_optimization_techniques(self) -> None:
        self.run_example_test("06_performance_optimization/optimization_techniques.py")
