from typing import List, Dict, Any
from src.avior.registry.dataset.base.preppers import IDatasetPrepper
from src.avior.registry.dataset.base.models import DatasetEntry


class CodePrepper(IDatasetPrepper):
    def get_required_keys(self) -> List[str]:
        return ["prompt", "tests", "language"]

    def create_dataset_entries(self, item: Dict[str, Any]) -> List[DatasetEntry]:
        return [
            DatasetEntry(
                query=item["prompt"],
                choices={},
                metadata={"tests": item["tests"], "language": item["language"]},
            )
        ]
