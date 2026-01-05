"""Test that our XCS return value fix works correctly.

This documents that we fixed the critical bug where @jit
was only returning the last computed value.
"""

import pytest

from ember.xcs import jit


class TestXCSReturnValueFix:
    """Verify the return value fix works."""

    def test_list_return_fixed(self):
        """Test that @jit now returns full lists."""

        @jit
        def build_list():
            a = "first"
            b = "second"
            c = "third"
            return [a, b, c]

        result = build_list()
        assert result == ["first", "second", "third"]

    def test_loop_return_fixed(self):
        """Test that @jit now handles loops correctly."""

        @jit
        def loop_list():
            results = []
            for i in range(3):
                results.append(f"item_{i}")
            return results

        result = loop_list()
        assert result == ["item_0", "item_1", "item_2"]

    def test_nested_structure_fixed(self):
        """Test that @jit preserves nested structures."""

        @jit
        def nested():
            outer = []
            inner1 = ["a", "b"]
            inner2 = ["c", "d"]
            outer.append(inner1)
            outer.append(inner2)
            return outer

        result = nested()
        assert result == [["a", "b"], ["c", "d"]]

    def test_dict_return(self):
        """Test that @jit handles dict returns."""

        @jit
        def build_dict():
            return {"a": 1, "b": 2, "c": 3}

        result = build_dict()
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_tuple_return(self):
        """Test that @jit handles tuple returns."""

        @jit
        def build_tuple():
            x = 10
            y = 20
            z = 30
            return (x, y, z)

        result = build_tuple()
        assert result == (10, 20, 30)

    def test_complex_nested_return(self):
        """Test complex nested data structures."""

        @jit
        def complex_data():
            data = {
                "lists": [[1, 2], [3, 4]],
                "dicts": {"inner": {"value": 42}},
                "mixed": [{"a": 1}, {"b": 2}],
            }
            return data

        result = complex_data()
        expected = {
            "lists": [[1, 2], [3, 4]],
            "dicts": {"inner": {"value": 42}},
            "mixed": [{"a": 1}, {"b": 2}],
        }
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
