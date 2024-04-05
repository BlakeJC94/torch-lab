"""Main data paths implemented as constants."""
from pathlib import Path

ROOT_DIR = Path(".")

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def get_task_dir_name(task) -> str:
    return f"{task.name}-{task.id}"
