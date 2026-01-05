from .test_base import ExampleTest


class TestDataProcessingExamples(ExampleTest):
    def test_loading_datasets(self) -> None:
        self.run_example_test("05_data_processing/loading_datasets.py")

    def test_streaming_data(self) -> None:
        self.run_example_test("05_data_processing/streaming_data.py")
