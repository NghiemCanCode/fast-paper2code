import datetime
import yaml
import os
from typing import Optional
from pathlib import Path


def load_yaml(yaml_file_path) -> Optional[dict]:
    with open(yaml_file_path, 'r', encoding="utf-8") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def unique_folder_path(folder_path, folder_name):
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path {folder_path} does not exist.")
    folder_name = f"{folder_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    return Path(folder_path) / folder_name