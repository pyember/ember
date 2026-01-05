from ember.api.record import Choice, ChoiceSet, DataRecord, TextContent
from scripts.ember_benchmark import normalize_prediction


def make_record(answer: str, choices: ChoiceSet | None = None) -> DataRecord:
    return DataRecord(
        question=TextContent(text="Question"),
        answer=TextContent(text=answer),
        choices=choices or ChoiceSet.empty(),
    )


def test_normalize_prediction_extracts_output_line() -> None:
    raw_text = "\n".join(
        [
            "Let me trace through this program step by step.",
            "The function examines subarrays and keeps track of the longest valid one.",
            "Output: 3",
            "Therefore the final answer is 3.",
        ]
    )
    record = make_record("3")
    assert normalize_prediction(raw_text, record, dataset_name="generic") == "3"


def test_normalize_prediction_respects_code_fence_literal() -> None:
    raw_text = "\n".join(
        [
            "Computation complete.",
            "The resulting value is:",
            "```",
            "10",
            "```",
        ]
    )
    record = make_record("10")
    assert normalize_prediction(raw_text, record, dataset_name="generic") == "10"


def test_normalize_prediction_preserves_quoted_literal() -> None:
    raw_text = "\n".join(
        [
            "After removing the trailing zeros we obtain:",
            "`'512301'`",
            "Hence the string returned is `'512301'`.",
        ]
    )
    record = make_record("'512301'")
    assert normalize_prediction(raw_text, record, dataset_name="generic") == "'512301'"


def test_normalize_prediction_keeps_multiple_choice_mapping() -> None:
    choices = ChoiceSet(
        (
            Choice(label="A", value="First option"),
            Choice(label="B", value="Second option"),
        )
    )
    record = make_record("Second option", choices=choices)
    assert normalize_prediction("B", record, dataset_name="generic") == "Second option"
