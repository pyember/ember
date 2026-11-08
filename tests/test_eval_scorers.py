"""Built-in scorers: one stated extraction rule each, no credit for a stray mention."""

import pytest

from ember.data import Example
from ember.eval import ExactMatch, MultipleChoice, NumericAnswer, Score, multiple_choice_prompt

ARITH = Example(question="What is 6 * 7?", answer="42")
CHOICE = Example(question="Pick the even one.", answer="C", choices=("1", "3", "4", "5"))


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("  42\n", Score(1.0, "42")),
        ("42.", Score(0.0, "42.")),
        ("The answer is 42", Score(0.0, "The answer is 42")),
    ],
)
def test_exact_match_ignores_only_surrounding_whitespace(output: str, expected: Score) -> None:
    assert ExactMatch()(ARITH, output) == expected


def test_exact_match_can_ignore_case() -> None:
    example = Example(question="Capital of France?", answer="Paris")
    assert ExactMatch()(example, "paris").value == 0.0
    assert ExactMatch(case_sensitive=False)(example, "PARIS").value == 1.0


@pytest.mark.parametrize(
    ("output", "value", "extracted"),
    [
        ("6 * 7 = 42", 1.0, "42"),
        ("It costs $42.00 in total.", 1.0, "42.00"),
        ("First 40, then 41, finally 42", 1.0, "42"),
        ("no digits here", 0.0, None),
        ("-42", 0.0, "-42"),
        ("6-7 is -1", 0.0, "-1"),
        ("7-6=1", 0.0, "1"),
    ],
)
def test_numeric_answer_reads_the_last_number(output: str, value: float, extracted: str) -> None:
    assert NumericAnswer()(ARITH, output) == Score(value, extracted)


def test_numeric_answer_gives_no_credit_for_a_number_mentioned_in_passing() -> None:
    # v0.1's NumericAnswerEvaluator counted any integer anywhere in the output.
    assert NumericAnswer()(ARITH, "It is not 42; it is 17.").value == 0.0
    assert NumericAnswer()(ARITH, " ".join(str(n) for n in range(100))).value == 0.0


def test_numeric_answer_compares_values_not_spellings() -> None:
    thousand = Example(question="q", answer="1,000")
    assert NumericAnswer()(thousand, "1000").value == 1.0
    assert NumericAnswer()(thousand, "1,000.0").value == 1.0
    assert NumericAnswer()(Example(question="q", answer="-3"), "x = -3").value == 1.0


def test_numeric_answer_after_a_marker_reads_the_first_number_after_its_last_use() -> None:
    scorer = NumericAnswer(after="####")
    assert scorer(ARITH, "#### 7 is wrong\n#### 42 (from 6 * 7)") == Score(1.0, "42")
    assert scorer(ARITH, "the result is 42") == Score(0.0, None)


def test_numeric_answer_rejects_a_non_numeric_reference() -> None:
    with pytest.raises(ValueError, match="needs a numeric answer"):
        NumericAnswer()(Example(question="q", answer="forty-two"), "42")


@pytest.mark.parametrize(
    ("output", "extracted"),
    [
        ("Let me think.\nAnswer: C", "C"),
        ("Answer: A\nOn reflection, Answer: (C)", "C"),
        ("**Answer:** **C**", "C"),
        ("answer:C", "C"),
        ("C", "C"),
        ("(C).", "C"),
        ("Answer: B", "B"),
        ("Answer: Because 4 is even", None),
        ("Answer: E", None),
        ("Answer: c", None),
        ("C is the even one", None),
        ("Answer: A careful reading gives C", None),
        ("Answer: C because 4 is even", None),
        ("Answer: C) 4", "C"),
        ("Answer: (C) 4", "C"),
        ("Answer: C. 4", "C"),
        ("Answer: C) 5", None),
        ("Answer: C) 4 because it is even", None),
        ("Answer: (C).\nThat is my final choice.", "C"),
        ("Reasoning.\r\nAnswer: C\r\n", "C"),
    ],
)
def test_multiple_choice_reads_the_last_answer_line(output: str, extracted: str | None) -> None:
    score = MultipleChoice()(CHOICE, output)
    assert score == Score(1.0 if extracted == "C" else 0.0, extracted)


def test_multiple_choice_needs_choices() -> None:
    with pytest.raises(ValueError, match="multiple-choice"):
        MultipleChoice()(ARITH, "Answer: A")


def test_the_prompt_asks_for_the_format_the_scorer_reads() -> None:
    prompt = multiple_choice_prompt(CHOICE)
    assert prompt.splitlines()[:6] == ["Pick the even one.", "", "A) 1", "B) 3", "C) 4", "D) 5"]
    assert '"Answer: X"' in prompt


def test_scores_outside_the_unit_interval_are_rejected() -> None:
    for value in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            Score(value, None)


def test_multiple_choice_does_not_read_a_word_as_a_letter() -> None:
    # "I" is a valid letter once there are nine or more choices.
    example = Example(question="q", answer="C", choices=tuple("0123456789"))
    assert MultipleChoice()(example, "Answer: I believe it is C").extracted is None
    assert MultipleChoice()(example, "Answer: I").extracted == "I"
