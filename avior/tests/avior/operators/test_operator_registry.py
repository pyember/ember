"""Tests for various operators and networks in the Avior framework."""

import pytest
from avior.core.non_graph import NoNGraph
from avior.registry import non
from avior.registry.dataset.dataset_registry import DatasetRegistry
from avior.registry.eval_function.eval_function_registry import (
    EvaluatorRegistry,
    MultipleChoiceEvaluator,
)
from avior.registry.model.model_registry import ModelRegistry
from avior.registry.operators.operator_registry import (
    OperatorFactory,
    OperatorRegistry,
    OperatorCode,
)
from avior.registry.operators.operator_base import (
    OperatorContext,
    NoNModule,
    NoNSequential,
)
from avior.registry.operators.operator_base import LMOperatorUnitConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union


@pytest.fixture(scope="module")
def setup_registries():
    """Set up and return the necessary registries for testing."""
    model_registry = ModelRegistry()
    operator_registry = OperatorRegistry()
    dataset_registry = DatasetRegistry()
    evaluator_registry = EvaluatorRegistry()
    evaluator_registry.register("multiple_choice", MultipleChoiceEvaluator())
    return model_registry, operator_registry, dataset_registry, evaluator_registry


@pytest.fixture(scope="module")
def load_questions(setup_registries):
    """Load and prepare test questions from the MMLU dataset."""
    _, _, dataset_registry, _ = setup_registries
    dataset_info_and_loader = dataset_registry.get("mmlu")
    _, dataset_loader_prepper = dataset_info_and_loader
    return dataset_loader_prepper.load_and_prepare(num_questions=20, subject="all")


def run_graph_test(
    graph_def: Union[List[str], NoNGraph],
    questions: List[OperatorContext],
    evaluator_registry: EvaluatorRegistry,
) -> float:
    """
    Run a test on a given graph definition with a set of questions.

    Args:
        graph_def: The graph definition to test, either as a list of strings or a NoNGraph object.
        questions: A list of OperatorContext objects representing the test questions.
        evaluator_registry: The registry containing evaluation functions.

    Returns:
        The accuracy of the model on the given questions as a percentage.
    """
    print(f"run_graph_test: Starting run_graph_test with graph_def: {graph_def}")
    model = (
        NoNGraph().parse_from_list(graph_def)
        if isinstance(graph_def, list)
        else graph_def
    )
    print(f"run_graph_test: Created model: {model}")
    total_score = 0

    def process_question(question: OperatorContext) -> float:
        """Process a single question and return the score."""
        print(f"run_graph_test: Processing question: {question.query[:50]}...")
        result = model.forward(question)
        print(f"run_graph_test: Forward pass result: {result}")

        final_answer = (
            result.final_answer if isinstance(result, OperatorContext) else result
        )
        print(f"run_graph_test: Final answer: {final_answer}")

        multiple_choice_evaluator = evaluator_registry.get("multiple_choice")
        extracted_answer = multiple_choice_evaluator.extract_answer(
            final_answer, question.choices
        )
        print(f"run_graph_test: Extracted answer: {extracted_answer}")

        evaluation_result = multiple_choice_evaluator.evaluate(
            extracted_answer, question.metadata["correct_answer"]
        )
        print(f"run_graph_test: Evaluation result: {evaluation_result}")
        return evaluation_result.score

    with ThreadPoolExecutor() as executor:
        future_to_question = {
            executor.submit(process_question, question): question
            for question in questions
        }
        for future in as_completed(future_to_question):
            total_score += future.result()

    return (total_score / len(questions)) * 100


class TestVerifierOperator:
    """Test suite for the Verifier operator."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries):
        """Set up the test environment for each test method."""
        # Extracting necessary components from setup_registries
        self.model_registry, _, _, _ = setup_registries

        # Creating a VerifierOperator instance using OperatorFactory with typed config
        unit_config = LMOperatorUnitConfig(model_name="gpt-4o", temperature=0.0)
        self.verifier = OperatorFactory.create(OperatorCode.VERIFIER, unit_config)
        print(f"Creating Verifier instance: {self.verifier}")

    def test_correct_answer(self):
        """Test the Verifier operator with a correct answer."""
        correct_input = OperatorContext(
            query="What is 2 + 2?",
            responses=["b"],
            choices={"a": "3", "b": "4", "c": "5"},
        )
        print(f"Testing with correct input: {correct_input}")
        result = self.verifier(correct_input)
        correct_result = result.metadata["verdict"]
        correct_answer = result.final_answer
        print(f"Correct result: {correct_result}, Correct answer: {correct_answer}")

        # Asserting verifier behavior for correct answer
        assert (
            correct_result == 1
        ), f"Verifier failed to identify correct answer. Got {correct_result}, expected 1."
        assert (
            correct_answer.lower() == "b"
        ), f"Verifier failed to extract correct answer. Got {correct_answer.lower()}, expected 'b'."

        print("Verifier operator test for correct answer completed successfully")

    def test_incorrect_answer(self):
        """Test the Verifier operator with an incorrect answer."""
        incorrect_input = OperatorContext(
            query="What is the capital of France?",
            responses=["The capital of France is London."],
            choices={"A": "Paris", "B": "London", "C": "Berlin"},
        )
        print(f"Testing with incorrect input: {incorrect_input}")
        result = self.verifier(incorrect_input)
        incorrect_result = result.metadata["verdict"]
        incorrect_answer = result.final_answer
        print(
            f"Incorrect result: {incorrect_result}, Incorrect answer: {incorrect_answer}"
        )

        # Asserting verifier behavior for incorrect answer
        assert (
            incorrect_result == 0
        ), f"Verifier failed to identify incorrect answer. Got {incorrect_result}, expected 0."
        assert incorrect_answer.lower() in [
            "a",
            "b",
            "c",
        ], f"Verifier failed to produce a valid answer type for incorrect input. Got {incorrect_answer.lower()}, expected one of ['a', 'b', 'c']."

        print("Verifier operator test for incorrect answer completed successfully")


@pytest.mark.parametrize(
    "operator_name, graph_def",
    [
        ("Ensemble", [["3:E"], "1:MC"]),
        ("JudgeBased", [["3:E"], "1:JB"]),
        ("VerifierBasedJudge", [["3:E"], "1:VBJ"]),
        ("GetAnswer", [["1:E"], "1:GA"]),
        ("MostCommon", [["3:E"], "1:MC"]),
        ("SelfRefinement", [["1:SR"], "1:GA"]),
        ("Discussion", [["1:D"], "1:GA"]),
        ("CompareResponses", [["2:E"], "1:CR"]),
    ],
)
def test_operator(operator_name, graph_def, setup_registries, load_questions):
    """
    Test various operators with different graph definitions.

    Args:
        operator_name: The name of the operator being tested.
        graph_def: The graph definition for the operator.
        setup_registries: Fixture providing necessary registries.
        load_questions: Fixture providing test questions.
    """
    _, _, _, evaluator_registry = setup_registries
    questions = load_questions

    accuracy = run_graph_test(graph_def, questions, evaluator_registry)

    assert (
        accuracy > 0
    ), f"{operator_name} operator failed to answer any questions correctly"
    print(f"{operator_name} operator accuracy: {accuracy}%")


def test_ensemble_operator_size(setup_registries, load_questions):
    """Test the Ensemble operator with different sizes."""
    _, _, _, evaluator_registry = setup_registries
    questions = load_questions

    # Testing different ensemble sizes
    for size in [1, 3, 5]:
        graph_def = [[f"{size}:E", "1:SR"], "1:MC"]
        accuracy = run_graph_test(graph_def, questions, evaluator_registry)
        assert (
            accuracy > 0
        ), f"Ensemble of size {size} failed to answer any questions correctly"
        print(f"Ensemble of size {size} accuracy: {accuracy}%")


class TestSimpleComplexNoN:
    """Test suite for simple and complex NoN graphs."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up the test environment for each test method."""
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions

        # Creating a sequential model using NoNGraph
        seq_model = NoNGraph()
        seq_model.add_node(
            "ensemble",
            OperatorFactory.create(
                OperatorCode.ENSEMBLE,
                LMOperatorUnitConfig(model_name="gpt-4o", count=3),
            ),
        )
        seq_model.add_node(
            "judge",
            OperatorFactory.create(
                OperatorCode.MOST_COMMON,
                LMOperatorUnitConfig(model_name="gpt-4o", count=1),
            ),
            inputs=["ensemble"],
        )

        # Creating a dict-like model using NoNGraph
        dict_model = NoNGraph()
        dict_model.add_node(
            "ensemble",
            OperatorFactory.create(
                OperatorCode.ENSEMBLE,
                LMOperatorUnitConfig(model_name="gpt-4o", count=3),
            ),
        )
        dict_model.add_node(
            "judge",
            OperatorFactory.create(
                OperatorCode.MOST_COMMON,
                LMOperatorUnitConfig(model_name="gpt-4o", count=1),
            ),
            inputs=["ensemble"],
        )

        # Creating a list-like model using NoNGraph
        list_model = NoNGraph()
        list_model.add_node(
            "ensemble1",
            OperatorFactory.create(
                "E", LMOperatorUnitConfig(model_name="gpt-4o", count=3)
            ),
        )
        list_model.add_node(
            "ensemble2",
            OperatorFactory.create(
                OperatorCode.ENSEMBLE,
                LMOperatorUnitConfig(model_name="gpt-4o", count=3),
            ),
        )
        list_model.add_node(
            "judge",
            OperatorFactory.create(
                OperatorCode.MOST_COMMON,
                LMOperatorUnitConfig(model_name="gpt-4o", count=1),
            ),
            inputs=["ensemble1", "ensemble2"],
        )

        # Creating the complex model
        self.complex_model = NoNGraph()
        self.complex_model.add_node("seq", seq_model)
        self.complex_model.add_node("dict", dict_model)
        self.complex_model.add_node("list", list_model)
        self.complex_model.add_node(
            "final",
            OperatorFactory.create(
                OperatorCode.MOST_COMMON,
                LMOperatorUnitConfig(model_name="gpt-4o", count=1),
            ),
            inputs=["seq", "dict", "list"],
        )

    def test_simple_complex_execution(self):
        """Test the execution of a simple complex NoNGraph."""
        print("Testing simple complex NoNGraph execution")
        accuracy = run_graph_test(
            self.complex_model, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Simple complex NoNGraph model failed to answer any questions correctly"
        print(f"Simple complex NoNGraph model accuracy: {accuracy}%")


class TestSimpleNestedGraph:
    """Test suite for simple nested graphs."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up the test environment for each test method."""
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions
        nested_graph = [
            ["ensemble1", "3:E:gpt-4o", []],
            ["ensemble2", "1:E:gpt-4o", []],
            [  # Nested subgraph
                ["sub_ensemble", "2:E:gpt-4o", []],
                ["sub_judge", "1:MC:gpt-4o", ["sub_ensemble"]],
            ],
            ["judge", "1:MC:gpt-4o", ["ensemble1", "ensemble2", "subgraph_2"]],
        ]
        self.nested_model = NoNGraph().parse_graph(nested_graph, named=True)

    def test_simple_nested_execution(self):
        """Test the execution of a simple nested graph."""
        print("Testing simple nested graph execution")
        accuracy = run_graph_test(
            self.nested_model, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Simple nested graph model failed to answer any questions correctly"
        print(f"Simple nested graph model accuracy: {accuracy}%")


class TestComplexNoNGraph:
    """Test suite for complex NoN graphs."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up the test environment for each test method."""
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions
        self.complex_graph_model = NoNGraph()
        self.complex_graph_model.add_node(
            "ensemble1",
            OperatorFactory.create("E", [{"model_name": "gpt-4o", "count": 3}]),
        )
        self.complex_graph_model.add_node(
            "ensemble2",
            OperatorFactory.create("E", [{"model_name": "gpt-4o", "count": 5}]),
        )
        self.complex_graph_model.add_node(
            "most_common",
            OperatorFactory.create("MC", [{"model_name": "gpt-4o", "count": 1}]),
            inputs=["ensemble1", "ensemble2"],
        )
        self.complex_graph_model.add_node(
            "judge",
            OperatorFactory.create("JB", [{"model_name": "gpt-4o", "count": 1}]),
            inputs=["most_common", "ensemble1", "ensemble2"],
        )
        self.complex_graph_model.add_node(
            "final",
            OperatorFactory.create("MC", [{"model_name": "gpt-4o", "count": 1}]),
            inputs=["judge"],
        )

    def test_complex_graph_execution(self):
        """Test the execution of a complex NoNGraph."""
        print("Testing complex NoNGraph execution")
        accuracy = run_graph_test(
            self.complex_graph_model, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Complex NoNGraph model failed to answer any questions correctly"
        print(f"Complex NoNGraph model accuracy: {accuracy}%")


class TestComplexNoNGraphDeclarative:
    """Test suite for complex NoN graphs defined declaratively."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up the test environment for each test method."""
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions

        graph_config = {
            "ensemble1": {"op": "E", "params": {"model_name": "gpt-4o", "count": 3}},
            "ensemble2": {"op": "E", "params": {"model_name": "gpt-4o", "count": 5}},
            "most_common": {
                "op": "MC",
                "params": {"model_name": "gpt-4o", "count": 1},
                "inputs": ["ensemble1", "ensemble2"],
            },
            "judge": {
                "op": "JB",
                "params": {"model_name": "gpt-4o", "count": 1},
                "inputs": ["most_common", "ensemble1", "ensemble2"],
            },
            "final": {
                "op": "MC",
                "params": {"model_name": "gpt-4o", "count": 1},
                "inputs": ["judge"],
            },
        }

        self.complex_graph_model = NoNGraph().parse_graph(graph_config, named=True)

    def test_complex_graph_declarative_execution(self):
        """Test the execution of a complex NoNGraph defined declaratively."""
        print("Testing complex NoNGraph declarative execution")
        accuracy = run_graph_test(
            self.complex_graph_model, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Complex NoNGraph declarative model failed to answer any questions correctly"
        print(f"Complex NoNGraph declarative model accuracy: {accuracy}%")


from typing import List, Tuple


class TestComplexNestedGraph:
    """Test suite for complex nested NoN graphs with multiple layers of subgraphs."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up a complex nested graph structure for testing.

        This method creates a nested graph with multiple subgraphs, ensembles,
        refiners, and judges to test intricate decision-making processes.
        """
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions
        complex_nested_graph: List[List] = [
            ["ensemble1", "5:E:gpt-4o", []],
            ["ensemble2", "3:E:gpt-4o", []],
            [  # Nested subgraph 1
                ["sub_ensemble1", "2:E:gpt-4o", []],
                ["sub_refiner1", "1:SR:gpt-4o", ["sub_ensemble1"]],
                ["sub_judge1", "1:MC:gpt-4o", ["sub_refiner1"]],
            ],
            [  # Nested subgraph 2
                ["sub_ensemble2", "3:E:gpt-4o", []],
                ["sub_refiner2", "2:SR:gpt-4o", ["sub_ensemble2"]],
                ["sub_judge2", "1:JB:gpt-4o", ["sub_refiner2"]],
            ],
            ["refiner", "2:SR:gpt-4o", ["ensemble1", "ensemble2"]],
            ["judge", "1:JB:gpt-4o", ["refiner", "subgraph_2", "subgraph_3"]],
            ["final", "1:MC:gpt-4o", ["judge"]],
        ]
        self.complex_nested_model = NoNGraph().parse_graph(
            complex_nested_graph, named=True
        )

    def test_complex_nested_execution(self):
        """Test the execution of a complex nested graph with multiple decision layers.

        This test evaluates the model's ability to process information through
        various nested subgraphs and make accurate decisions based on the
        combined outputs of these subgraphs.
        """
        print("Testing complex nested graph execution")
        accuracy = run_graph_test(
            self.complex_nested_model, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Complex nested graph model failed to answer any questions correctly"
        print(f"Complex nested graph model accuracy: {accuracy}%")


class TestComplexNoNSequential:
    """Test suite for complex NoNSequential models with multiple sequential operators."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up a complex sequential model with ensemble, self-refinement, and majority choice.

        This setup creates a sequential model that processes inputs through multiple
        stages, including an ensemble of models, self-refinement, and final decision making.
        """
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions
        self.complex_seq_model = NoNSequential(
            OperatorFactory.create(
                OperatorCode.ENSEMBLE,
                LMOperatorUnitConfig(model_name="gpt-4o", count=5),
            ),
            OperatorFactory.create(
                OperatorCode.SELF_REFINEMENT,
                LMOperatorUnitConfig(model_name="gpt-4o", count=2),
            ),
            OperatorFactory.create(
                OperatorCode.MOST_COMMON,
                LMOperatorUnitConfig(model_name="gpt-4o", count=1),
            ),
        )

    def test_complex_sequential_execution(self):
        """Test the execution of a complex NoNSequential model with multiple processing stages.

        This test evaluates the model's ability to process information sequentially
        through an ensemble, refine the results, and make a final decision based on
        majority choice.
        """
        print("Testing complex NoNSequential execution")
        accuracy = run_graph_test(
            self.complex_seq_model, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Complex NoNSequential model failed to answer any questions correctly"
        print(f"Complex NoNSequential model accuracy: {accuracy}%")


class TestSimpleNoNSequential:
    """Test suite for simple NoNSequential models with basic sequential processing."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up a simple sequential model with an ensemble followed by majority choice.

        This setup creates a basic sequential model that processes inputs through
        an ensemble of models and then makes a decision based on majority choice.
        """
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions
        self.seq_model = NoNSequential(
            OperatorFactory.create(
                OperatorCode.ENSEMBLE,
                LMOperatorUnitConfig(model_name="gpt-4o", count=3),
            ),
            OperatorFactory.create(
                OperatorCode.MOST_COMMON,
                LMOperatorUnitConfig(model_name="gpt-4o", count=1),
            ),
        )

    def test_simple_sequential_execution(self):
        """Test the execution of a simple NoNSequential model with basic processing.

        This test evaluates the model's ability to process information through
        a simple ensemble and make decisions based on majority choice, assessing
        its effectiveness in basic sequential processing.
        """
        print("Testing simple NoNSequential execution")
        accuracy = run_graph_test(
            self.seq_model, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Simple NoNSequential model failed to answer any questions correctly"
        print(f"Simple NoNSequential model accuracy: {accuracy}%")


class TestSimpleNoNModuleNetwork:
    """Test suite for simple NoNModule networks with basic ensemble and judge components."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up a simple NoNModule network with an ensemble and a judge.

        This setup creates a basic network that processes inputs through an
        ensemble of models and then uses a judge to make the final decision.
        """
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions

        class SimpleNetwork(NoNModule):
            """A simple network using NoNModule with ensemble and judge components."""

            def __init__(self):
                """Initialize the SimpleNetwork with an ensemble and a judge."""
                super().__init__()
                self.ensemble = non.Ensemble(num_units=3, model_name="gpt-4o")
                self.judge = non.Judge(model_name="gpt-4o")

            def forward(self, input_data: OperatorContext) -> OperatorContext:
                """Process input through the ensemble and judge.

                Args:
                    input_data: The input data containing the question and context.

                Returns:
                    The final output after processing through the ensemble and judge.
                """
                ensemble_output = self.ensemble(input_data)
                final_output = self.judge(ensemble_output)
                return final_output

        self.simple_network = SimpleNetwork()

    def test_simple_non_module_network(self):
        """Test the execution of a simple NoNModule network with basic components.

        This test evaluates the model's ability to process information through
        a simple ensemble and make decisions using a judge, assessing its
        effectiveness in basic modular processing.
        """
        print("Testing simple NoNModule network execution")
        accuracy = run_graph_test(
            self.simple_network, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Simple NoNModule network failed to answer any questions correctly"
        print(f"Simple NoNModule network accuracy: {accuracy}%")


class TestComplexNoNModuleNetwork:
    """Test suite for complex NoNModule networks with multiple components and decision layers."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up a complex NoNModule network with multiple ensembles, a refiner, and a judge.

        This setup creates an advanced network that processes inputs through multiple
        ensembles, refines the results, and uses a judge for final decision making.
        """
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions

        class ComplexNetwork(NoNModule):
            """A complex network using NoNModule with multiple processing stages."""

            def __init__(self):
                """Initialize the ComplexNetwork with multiple components."""
                super().__init__()
                ensemble_config1 = LMOperatorUnitConfig(model_name="gpt-4o", count=5)
                ensemble_config2 = LMOperatorUnitConfig(model_name="gpt-4o", count=3)
                refiner_config = LMOperatorUnitConfig(model_name="gpt-4o", count=1)
                judge_config = LMOperatorUnitConfig(model_name="gpt-4o", count=1)

                self.ensemble1 = non.Ensemble(ensemble_config1)
                self.ensemble2 = non.Ensemble(ensemble_config2)
                self.refiner = non.SelfRefinement(refiner_config)
                self.judge = non.Judge(judge_config)

            def forward(self, input_data: OperatorContext) -> OperatorContext:
                """Process input through multiple ensembles, refiner, and judge.

                Args:
                    input_data: The input data containing the question and context.

                Returns:
                    The final output after processing through all components.
                """
                ensemble1_output = self.ensemble1(input_data)
                ensemble2_output = self.ensemble2(input_data)
                refined_output = self.refiner(ensemble1_output)
                final_output = self.judge(
                    OperatorContext(
                        query=input_data.query,
                        responses=[
                            f"Ensemble 1: {ensemble1_output.final_answer}\n",
                            f"Ensemble 2: {ensemble2_output.final_answer}\n",
                            f"Refined: {refined_output.final_answer}",
                        ],
                    )
                )
                return final_output

        self.complex_network = ComplexNetwork()

    def test_complex_non_module_network(self):
        """Test the execution of a complex NoNModule network with multiple processing stages.

        This test evaluates the model's ability to process information through
        multiple ensembles, refine the results, and make decisions using a judge,
        assessing its effectiveness in complex modular processing.
        """
        print("Testing complex NoNModule network execution")
        accuracy = run_graph_test(
            self.complex_network, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Complex NoNModule network failed to answer any questions correctly"
        print(f"Complex NoNModule network accuracy: {accuracy}%")


class TestNestedNoNModuleNetwork:
    """Test suite for nested NoNModule networks with hierarchical decision-making."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up a nested NoNModule network with subnetworks and a main network.

        This setup creates a hierarchical network structure with subnetworks that
        process inputs independently, and a main network that combines their outputs.
        """
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions

        class SubNetwork(NoNModule):
            """A subnetwork using NoNModule for initial processing."""

            def __init__(self):
                """Initialize the SubNetwork with an ensemble and a refiner."""
                super().__init__()
                self.ensemble = non.Ensemble(num_units=2, model_name="gpt-4o")
                self.refiner = non.SelfRefinement(model_name="gpt-4o")

            def forward(self, input_data: OperatorContext) -> OperatorContext:
                """Process input through the ensemble and refiner.

                Args:
                    input_data: The input data for the subnetwork.

                Returns:
                    The refined output after processing through the subnetwork.
                """
                ensemble_output = self.ensemble(input_data)
                refined_output = self.refiner(ensemble_output)
                return refined_output

        class NestedNetwork(NoNModule):
            """A nested network using NoNModule with subnetworks and additional components."""

            def __init__(self):
                """Initialize the NestedNetwork with subnetworks and additional components."""
                super().__init__()
                self.subnetwork1 = SubNetwork()
                self.subnetwork2 = SubNetwork()
                self.ensemble = non.Ensemble(num_units=3, model_name="gpt-4o")
                self.judge = non.Judge(model_name="gpt-4o")

            def forward(self, input_data: OperatorContext) -> OperatorContext:
                """Process input through subnetworks, ensemble, and judge.

                Args:
                    input_data: The input data for the network.

                Returns:
                    The final output after processing through all components.
                """
                sub1_output = self.subnetwork1(input_data)
                sub2_output = self.subnetwork2(input_data)
                ensemble_output = self.ensemble(input_data)
                final_output = self.judge(
                    OperatorContext(
                        query=input_data.query,
                        responses=[
                            f"Subnetwork 1: {sub1_output.final_answer}\n"
                            f"Subnetwork 2: {sub2_output.final_answer}\n"
                            f"Ensemble: {ensemble_output.final_answer}"
                        ],
                    )
                )
                return final_output

        self.nested_network = NestedNetwork()

    def test_nested_non_module_network(self):
        """Test the execution of a nested NoNModule network with hierarchical processing.

        This test evaluates the model's ability to process information through
        multiple layers of subnetworks and make decisions based on their combined
        outputs, assessing its effectiveness in hierarchical decision-making.
        """
        print("Testing nested NoNModule network execution")
        accuracy = run_graph_test(
            self.nested_network, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Nested NoNModule network failed to answer any questions correctly"
        print(f"Nested NoNModule network accuracy: {accuracy}%")


class TestCustomNoNModuleNetwork:
    """Test suite for custom NoNModule networks with specialized components."""

    @pytest.fixture(autouse=True)
    def setup(self, setup_registries, load_questions):
        """Set up a custom NoNModule network with specialized components and configurations.

        This setup creates a network with custom-configured ensemble, verifier, and judge
        components to test specialized processing strategies.
        """
        self.evaluator_registry = setup_registries[3]
        self.questions = load_questions

        class CustomNetwork(NoNModule):
            """A custom network using NoNModule with specialized components and configurations."""

            def __init__(self):
                """Initialize the CustomNetwork with specially configured components."""
                super().__init__()
                self.ensemble = non.Ensemble(
                    num_units=4, model_name="gpt-4o", temperature=0.7
                )
                self.verifier = non.Verifier(model_name="gpt-4o", temperature=0.5)
                self.judge = non.Judge(model_name="gpt-4o", temperature=0.3)

            def forward(self, input_data: OperatorContext) -> OperatorContext:
                """Process input through custom-configured ensemble, verifier, and judge.

                Args:
                    input_data: The input data for the network.

                Returns:
                    The final output after processing through all components.
                """
                ensemble_output = self.ensemble(input_data)
                verified_result = self.verifier(ensemble_output)
                final_output = self.judge(
                    OperatorContext(
                        query=input_data.query,
                        responses=[
                            f"Ensemble: {ensemble_output.final_answer}\n"
                            f"Verified: {verified_result[1]}"
                        ],
                    )
                )
                return final_output

        self.custom_network = CustomNetwork()

    def test_custom_non_module_network(self):
        """Test the execution of a custom NoNModule network with specialized components.

        This test evaluates the model's ability to process information through
        custom-configured components, assessing its effectiveness in specialized
        decision-making scenarios with varying temperatures and verification steps.
        """
        print("Testing custom NoNModule network execution")
        accuracy = run_graph_test(
            self.custom_network, self.questions, self.evaluator_registry
        )
        assert (
            accuracy > 0
        ), "Custom NoNModule network failed to answer any questions correctly"
        print(f"Custom NoNModule network accuracy: {accuracy}%")
