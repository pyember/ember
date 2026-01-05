from .test_base import ExampleTest


class TestSimplifiedApisExamples(ExampleTest):
    def test_model_binding_patterns(self) -> None:
        self.run_example_test("03_simplified_apis/model_binding_patterns.py")

    def test_natural_api_showcase(self) -> None:
        self.run_example_test("03_simplified_apis/natural_api_showcase.py")

    def test_simplified_workflows(self) -> None:
        self.run_example_test("03_simplified_apis/simplified_workflows.py")

    def test_zero_config_jit(self) -> None:
        self.run_example_test("03_simplified_apis/zero_config_jit.py")
