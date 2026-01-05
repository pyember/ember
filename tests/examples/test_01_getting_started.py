from .test_base import ExampleTest


class TestGettingStartedExamples(ExampleTest):
    def test_hello_world(self) -> None:
        self.run_example_test("01_getting_started/hello_world.py")

    def test_first_model_call(self) -> None:
        self.run_example_test("01_getting_started/first_model_call.py")

    def test_basic_prompt_engineering(self) -> None:
        self.run_example_test("01_getting_started/basic_prompt_engineering.py")

    def test_model_comparison(self) -> None:
        self.run_example_test("01_getting_started/model_comparison.py")
