from .test_base import ExampleTest


class TestCompoundAIExamples(ExampleTest):
    def test_judge_synthesis(self) -> None:
        self.run_example_test("04_compound_ai/judge_synthesis.py")

    def test_operators_progressive_disclosure(self) -> None:
        self.run_example_test("04_compound_ai/operators_progressive_disclosure.py")

    def test_simple_ensemble(self) -> None:
        self.run_example_test("04_compound_ai/simple_ensemble.py")

    def test_specifications_progressive(self) -> None:
        self.run_example_test("04_compound_ai/specifications_progressive.py")
