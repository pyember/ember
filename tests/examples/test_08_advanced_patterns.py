from .test_base import ExampleTest


class TestAdvancedPatternsExamples(ExampleTest):
    def test_advanced_techniques(self) -> None:
        self.run_example_test("08_advanced_patterns/advanced_techniques.py")

    def test_jax_xcs_integration(self) -> None:
        self.run_example_test("08_advanced_patterns/jax_xcs_integration.py")
