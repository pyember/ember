from __future__ import annotations

from ember.utils.eval.extractors import FinalLetterExtractor


def test_final_letter_handles_markdown_and_colon():
    extractor = FinalLetterExtractor(valid_letters="ABCD")

    assert extractor.extract("Answer: **B**. Because...") == "B"
    assert extractor.extract("final answer is (c)") == "C"


def test_final_letter_understands_statement_and_option_phrasing():
    extractor = FinalLetterExtractor(valid_letters="WXYZ")

    assert extractor.extract("The statement is X") == "X"
    assert extractor.extract("option is Y because reasons") == "Y"


def test_final_letter_accepts_single_letter_line():
    extractor = FinalLetterExtractor(valid_letters="ABCD")

    assert extractor.extract("B\n\nReason: ...") == "B"


def test_final_letter_rejects_multi_letter_lists():
    extractor = FinalLetterExtractor(valid_letters="ABCDEFG")

    assert extractor.extract("Answer: A and E") == ""
    assert extractor.extract("A, C, D") == ""
