import threading
import unittest
import dataclasses
from typing import List

from ember.core.registry.operator.base._module import EmberModule, ember_field


@dataclasses.dataclass(frozen=True, init=True)
class ConcurrencyTestModule(EmberModule):
    """Immutable EmberModule extension for concurrency testing.

    Attributes:
        counter (int): A counter used to verify thread-safe access.
    """

    counter: int = ember_field()


class TestConcurrency(unittest.TestCase):
    """Unit tests to verify thread safety of ConcurrencyTestModule."""

    def test_thread_safety(self) -> None:
        """Verifies that concurrent reads to an immutable module's field are thread-safe.

        This test creates 100 threads that each read the 'counter' attribute from a
        ConcurrencyTestModule instance initialized with 0. It confirms that every thread
        retrieves the correct value (0), thereby ensuring no race conditions occur.
        """
        module_instance: ConcurrencyTestModule = ConcurrencyTestModule(counter=0)
        results: List[int] = []

        def read_counter() -> None:
            """Reads the 'counter' value from module_instance and appends it to results."""
            results.append(module_instance.counter)

        thread_list: List[threading.Thread] = [
            threading.Thread(target=read_counter) for _ in range(100)
        ]
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()

        self.assertEqual(
            first=len(results),
            second=100,
            msg="Expected exactly 100 results from concurrent reads.",
        )
        self.assertTrue(
            all(val == 0 for val in results), msg="All counter values should be 0."
        )


if __name__ == "__main__":
    unittest.main()
