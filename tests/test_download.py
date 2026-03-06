import os
import sys
import unittest
from unittest.mock import MagicMock, patch

def get_download_function():
    # Path to the script relative to the repo root
    script_path = os.path.join(os.path.dirname(__file__), '..', 'yolov4', 'deeptrash_yolov4.py')

    with open(script_path, "r") as f:
        lines = f.readlines()

    func_lines = []
    in_func = False
    for line in lines:
        if line.startswith("def download("):
            in_func = True
            func_lines.append(line)
        elif in_func:
            # End of function detection based on indentation
            # It starts with "def download(path):", followed by indented lines
            if line.strip() != "" and not line.startswith(" ") and not line.startswith("\t") and not line.startswith("#"):
                break
            func_lines.append(line)

    func_code = "".join(func_lines)

    # Execute the extracted function code in a fresh namespace
    namespace = {}
    exec(func_code, namespace)

    return namespace["download"]

class TestDownloadFunction(unittest.TestCase):
    def test_download_called_with_correct_path(self):
        # Extract the function
        download_func = get_download_function()

        # Setup mocks for google.colab
        mock_google = MagicMock()
        mock_colab = MagicMock()
        mock_files = MagicMock()

        mock_colab.files = mock_files
        mock_google.colab = mock_colab

        # We need to mock sys.modules because the function does:
        # from google.colab import files
        with patch.dict("sys.modules", {"google": mock_google, "google.colab": mock_colab}):
            test_path = "dummy/path/to/file.zip"
            download_func(test_path)

            # Assert that files.download was called with our test_path
            mock_files.download.assert_called_once_with(test_path)

if __name__ == "__main__":
    unittest.main()
