from .test_base import ExampleTest


class TestCoreConceptsExamples(ExampleTest):
    def test_context_management(self) -> None:
        self.run_example_test("02_core_concepts/context_management.py")

    def test_error_handling(self) -> None:
        self.run_example_test("02_core_concepts/error_handling.py")

    def test_operators_basics(self) -> None:
        self.run_example_test("02_core_concepts/operators_basics.py")

    def test_rich_specifications(self) -> None:
        self.run_example_test("02_core_concepts/rich_specifications.py")

    def test_type_safety(self) -> None:
        self.run_example_test("02_core_concepts/type_safety.py")
