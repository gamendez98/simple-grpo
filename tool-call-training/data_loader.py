import os


def find_files_by_name(root_path: str, target_name: str):
    """
    Recursively yields paths of files in `root_path` (and subdirectories)
    that have the given `target_name`.
    """
    for entry in os.scandir(root_path):
        if entry.is_file() and entry.name == target_name:
            yield entry.path
        elif entry.is_dir():
            yield from find_files_by_name(entry.path, target_name)

class DataLoader:
    EXPECTED_FILE_NAME = "annotated_conversations.json"

    def __init__(self, data_path: str):
        self.data_path = data_path

    def iterate_annotated_conversations(self):
        yield from find_files_by_name(self.data_path, self.EXPECTED_FILE_NAME)

    def