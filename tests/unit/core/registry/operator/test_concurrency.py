import threading
import unittest
import dataclasses
from src.ember.core.registry.operator.base._module import EmberModule, ember_field

@dataclasses.dataclass(frozen=True, init=True)
class ConcurrencyTestModule(EmberModule):
    counter: int = ember_field()

class TestConcurrency(unittest.TestCase):
    def test_thread_safety(self) -> None:
        instance = ConcurrencyTestModule(counter=0)

        results = []

        def read_counter():
            results.append(instance.counter)

        threads = [threading.Thread(target=read_counter) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 100)
        self.assertTrue(all(val == 0 for val in results))

if __name__ == "__main__":
    unittest.main() 