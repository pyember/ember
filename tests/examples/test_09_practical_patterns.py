from .test_base import ExampleTest


class TestPracticalPatternsExamples(ExampleTest):
    def test_chain_of_thought(self) -> None:
        self.run_example_test("09_practical_patterns/chain_of_thought.py")

    def test_rag_pattern(self) -> None:
        self.run_example_test("09_practical_patterns/rag_pattern.py")

    def test_structured_output(self) -> None:
        self.run_example_test("09_practical_patterns/structured_output.py")
