import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import re
import os

# Dynamically extract the imShow function from the unimportable colab script
def extract_code():
    filepath = os.path.join(os.path.dirname(__file__), '../../yolov4/deeptrash_yolov4.py')
    with open(filepath, 'r') as f:
        content = f.read()

    match = re.search(r'(def imShow\(path\):.*?plt\.show\(\))', content, re.DOTALL)
    if match:
        return match.group(1)
    return None

code = extract_code()
if code:
    exec(code, globals())
else:
    raise RuntimeError("Failed to extract imShow function from deeptrash_yolov4.py")

class TestImShow:
    @patch('cv2.imread')
    @patch('cv2.resize')
    @patch('cv2.cvtColor')
    @patch('matplotlib.pyplot.gcf')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.show')
    def test_imshow_happy_path(self, mock_show, mock_imshow, mock_axis, mock_gcf, mock_cvtColor, mock_resize, mock_imread):
        # Setup mocks
        mock_image = np.zeros((100, 200, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        mock_resized_image = np.zeros((300, 600, 3), dtype=np.uint8)
        mock_resize.return_value = mock_resized_image

        mock_converted_image = np.zeros((300, 600, 3), dtype=np.uint8)
        mock_cvtColor.return_value = mock_converted_image

        mock_fig = MagicMock()
        mock_gcf.return_value = mock_fig

        # Call the function
        imShow('dummy/path/to/image.jpg')

        import cv2

        # Verify imread was called correctly
        mock_imread.assert_called_once_with('dummy/path/to/image.jpg')

        # Verify resize was called with correct arguments
        mock_resize.assert_called_once()
        args, kwargs = mock_resize.call_args
        assert args[0] is mock_image
        assert args[1] == (600, 300) # (3*width, 3*height)
        assert kwargs['interpolation'] == cv2.INTER_CUBIC

        # Verify plot setup
        mock_gcf.assert_called_once()
        mock_fig.set_size_inches.assert_called_once_with(18, 10)
        mock_axis.assert_called_once_with("off")

        # Verify color conversion
        mock_cvtColor.assert_called_once()
        args, kwargs = mock_cvtColor.call_args
        assert args[0] is mock_resized_image
        assert args[1] == cv2.COLOR_BGR2RGB

        # Verify show
        mock_imshow.assert_called_once_with(mock_converted_image)
        mock_show.assert_called_once()

    @patch('cv2.imread')
    def test_imshow_file_not_found(self, mock_imread):
        # When cv2.imread fails to read an image (e.g. file not found), it returns None
        mock_imread.return_value = None

        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'shape'"):
            imShow('non_existent_image.jpg')
