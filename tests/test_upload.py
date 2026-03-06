import re
import sys
import pytest
from unittest.mock import patch, mock_open

# Extract the upload function from the script to avoid syntax errors from colab magics
def get_upload_func():
    try:
        with open('yolov4/deeptrash_yolov4.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        # For pytest running from a different working directory
        with open('../yolov4/deeptrash_yolov4.py', 'r', encoding='utf-8') as f:
            content = f.read()

    match = re.search(r'def upload\(\):[\s\S]*?(?=\n\S|\Z)', content)
    if match:
        namespace = {}
        exec(match.group(0), namespace)
        return namespace['upload']
    raise Exception("upload function not found in yolov4/deeptrash_yolov4.py")

upload = get_upload_func()

@patch('builtins.print')
def test_upload_single_file(mock_print):
    """Test uploading a single file."""
    # Mocking google.colab.files.upload
    colab_mock = type('colab_mock', (), {})
    files_mock = type('files_mock', (), {})
    files_mock.upload = lambda: {'test_image.jpg': b'fake image data'}
    colab_mock.files = files_mock
    sys.modules['google.colab'] = colab_mock
    sys.modules['google'] = type('google', (), {})

    with patch('builtins.open', mock_open()) as mocked_file:
        upload()

        # Verify the file was opened with correct name and mode
        mocked_file.assert_called_once_with('test_image.jpg', 'wb')

        # Verify the data was written
        mocked_file().write.assert_called_once_with(b'fake image data')

        # Verify print statement was called
        mock_print.assert_called_once_with('saved file', 'test_image.jpg')


@patch('builtins.print')
def test_upload_multiple_files(mock_print):
    """Test uploading multiple files at once."""
    files_data = {
        'img1.jpg': b'data 1',
        'img2.png': b'data 2',
        'document.txt': b'data 3'
    }

    # Mocking google.colab.files.upload
    colab_mock = type('colab_mock', (), {})
    files_mock = type('files_mock', (), {})
    files_mock.upload = lambda: files_data
    colab_mock.files = files_mock
    sys.modules['google.colab'] = colab_mock
    sys.modules['google'] = type('google', (), {})

    with patch('builtins.open', mock_open()) as mocked_file:
        upload()

        # Should be called 3 times
        assert mocked_file.call_count == 3

        # Verify calls to open
        mocked_file.assert_any_call('img1.jpg', 'wb')
        mocked_file.assert_any_call('img2.png', 'wb')
        mocked_file.assert_any_call('document.txt', 'wb')

        # Verify calls to print
        assert mock_print.call_count == 3
        mock_print.assert_any_call('saved file', 'img1.jpg')
        mock_print.assert_any_call('saved file', 'img2.png')
        mock_print.assert_any_call('saved file', 'document.txt')


@patch('builtins.print')
def test_upload_empty(mock_print):
    """Test when no files are uploaded."""
    # Mocking google.colab.files.upload returning empty dict
    colab_mock = type('colab_mock', (), {})
    files_mock = type('files_mock', (), {})
    files_mock.upload = lambda: {}
    colab_mock.files = files_mock
    sys.modules['google.colab'] = colab_mock
    sys.modules['google'] = type('google', (), {})

    with patch('builtins.open', mock_open()) as mocked_file:
        upload()

        # Should not be called
        mocked_file.assert_not_called()
        mock_print.assert_not_called()
