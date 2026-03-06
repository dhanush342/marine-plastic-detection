import tempfile
import os

# Read the file
with open('yolov4/deeptrash_yolov4.py', 'r') as f:
    source = f.read()

# Since the file has IPython magics, parsing it with ast directly will fail.
# Instead, we will find the function definition using basic string manipulation.
lines = source.split('\n')
func_lines = []
in_func = False
for line in lines:
    if line.startswith('def writetemplate(line, cell):'):
        in_func = True
        func_lines.append(line)
    elif in_func:
        # Stop at the first non-empty, non-indented line after the function starts
        if not line.startswith(' ') and line != '':
            break
        func_lines.append(line)

func_code = '\n'.join(func_lines)
exec(func_code)

def test_writetemplate():
    """
    Test the writetemplate function that replaces string templates
    with values from the global context.
    """
    # Create a temporary file to hold the output
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tmp:
        tmp_path = tmp.name

    try:
        # Define some global variables for the template
        # The writetemplate function uses **globals(), so we need variables in global scope
        global steps_str, num_filters, num_classes
        steps_str = "8000,9000"
        num_filters = 255
        num_classes = 80

        # Test content with format strings
        cell_content = """[net]
steps={steps_str}
filters={num_filters}
classes={num_classes}
"""
        # Call the dynamically loaded function (it was defined by exec)
        writetemplate(tmp_path, cell_content)  # type: ignore

        # Verify the file was created and content matches
        assert os.path.exists(tmp_path)
        with open(tmp_path, 'r') as f:
            content = f.read()

        assert "steps=8000,9000" in content
        assert "filters=255" in content
        assert "classes=80" in content

        # We can also verify an explicit expected output
        expected_output = """[net]
steps=8000,9000
filters=255
classes=80
"""
        assert content == expected_output

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
