from typing import List, Dict, Any, Optional
from src.avior.registry.dataset.base.preppers import IDatasetPrepper
from src.avior.registry.dataset.base.models import DatasetEntry
from src.avior.registry.dataset.base.config import BaseDatasetConfig

class MMLUConfig(BaseDatasetConfig):
    """
    Represents MMLU-specific config fields, e.g. a sub-dataset name on HF.
    """
    config_name: Optional[str] = None
    split: Optional[str] = None

class MMLUPrepper(IDatasetPrepper):
    def __init__(self, config: MMLUConfig = MMLUConfig()):
        super().__init__(config)
        self.config_name = self._config.config_name
        self.split = self._config.split

    def get_required_keys(self) -> List[str]:
        # Removed "subject" to handle possible missing subject more gracefully
        return ["question", "choices", "answer"]

    def create_dataset_entries(self, item: Dict[str, Any]) -> List[DatasetEntry]:
        question = item["question"]
        # 'choices' is presumably a list of strings, e.g. ["Alpha", "Beta", "Gamma", "Delta"]
        choices = {chr(65 + i): choice for i, choice in enumerate(item["choices"])}

        # Handle 'answer' as either int or single-letter string
        raw_answer = item["answer"]
        if isinstance(raw_answer, int):
            correct_answer = chr(65 + raw_answer)
        else:
            correct_answer = raw_answer

        # Handle 'subject' gracefully if it is missing
        subject = item.get("subject", None)

        return [
            DatasetEntry(
                query=question,
                choices=choices,
                metadata={
                    "correct_answer": correct_answer,
                    "subject": subject,
                    "config_name": self.config_name,
                },
            )
        ]
