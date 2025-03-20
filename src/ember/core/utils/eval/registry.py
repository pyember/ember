from typing import Any, Callable, Dict

from .base_evaluator import IEvaluator


class EvaluatorRegistry:
    """Registry for storing and retrieving evaluator factories by a unique name.

    Evaluator factories should be callables that return an instance of IEvaluator.

    Methods:
        register: Registers an evaluator factory with a unique name.
        create: Creates an evaluator instance using the registered factory.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., IEvaluator[Any, Any]]] = {}

    def register(self, name: str, factory: Callable[..., IEvaluator[Any, Any]]) -> None:
        """Registers a factory callable under the given name.

        Args:
            name (str): A unique identifier for the evaluator.
            factory (Callable[..., IEvaluator[Any, Any]]): A callable that returns an IEvaluator instance.
        """
        self._registry[name] = factory

    def create(self, name: str, **kwargs: Any) -> IEvaluator[Any, Any]:
        """Creates an evaluator instance using the registered factory.

        Args:
            name (str): The unique identifier for the evaluator.
            **kwargs: Additional keyword arguments for the factory.

        Returns:
            IEvaluator[Any, Any]: An instance of the requested evaluator.

        Raises:
            KeyError: If no evaluator is registered under the specified name.
        """
        if name not in self._registry:
            raise KeyError(f"No evaluator registered with name: {name}")
        return self._registry[name](**kwargs)
