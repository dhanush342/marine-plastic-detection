import os
import re

def extract_writetemplate(filepath):
    """
    Extracts the writetemplate function using string matching, since the file
    contains invalid python syntax (IPython magics, markdown blocks, etc.).
    """
    with open(filepath, 'r') as f:
        source = f.read()

    # Find the function definition
    match = re.search(r'(@register_line_cell_magic\s+def writetemplate\(line, cell\):.*?)(?:\n\S|$)', source, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find writetemplate in {filepath}")

    func_str = match.group(1)

    # In order to test it, we need a mock for register_line_cell_magic
    mock_setup = """
def register_line_cell_magic(func):
    return func
"""
    return mock_setup + func_str

def test_yolov5_writetemplate():
    """
    Tests that yolov5 writetemplate only formats safe globals and prevents code/variable injection.
    """
    func_code = extract_writetemplate('yolov5/deeptrash_yolov5.py')

    # We create a new local environment to execute the extracted function
    test_env = {}
    exec(func_code, test_env)
    writetemplate = test_env['writetemplate']

    # Create test cell and line
    test_line = 'test_yolov5_output.yaml'
    # Try to access a sensitive global variable through formatting
    test_cell = "Safe: {num_classes}, Unsafe: {sensitive_data}"

    # Set up our testing globals
    # We need to monkeypatch globals() within the executed function's scope,
    # or ensure our test environment exposes what we need.
    # Since the extracted function calls `globals()`, it will look at test_env.
    test_env['num_classes'] = '80'
    test_env['sensitive_data'] = 'secret_key_123'

    # Because `globals()` inside the function will refer to `test_env` (the module dict it was executed in),
    # formatting should only successfully resolve `num_classes` and raise KeyError for `sensitive_data`
    # if our fix is correct. If it used `**globals()`, it would succeed and output the secret.

    try:
        writetemplate(test_line, test_cell)
        assert False, "Should have raised KeyError for sensitive_data"
    except KeyError as e:
        assert 'sensitive_data' in str(e) or e.args[0] == 'sensitive_data'

    # Clean up
    if os.path.exists(test_line):
        os.remove(test_line)

def test_yolov4_writetemplate():
    """
    Tests that yolov4 writetemplate only formats safe globals and prevents code/variable injection.
    """
    func_code = extract_writetemplate('yolov4/deeptrash_yolov4.py')

    test_env = {}
    exec(func_code, test_env)
    writetemplate = test_env['writetemplate']

    test_line = 'test_yolov4_output.cfg'
    test_cell = "Safe: {num_classes}, {steps_str}, {num_filters}, Unsafe: {sensitive_data}"

    test_env['num_classes'] = '3'
    test_env['steps_str'] = '4800,5400'
    test_env['num_filters'] = '24'
    test_env['sensitive_data'] = 'secret_key_456'

    try:
        writetemplate(test_line, test_cell)
        assert False, "Should have raised KeyError for sensitive_data"
    except KeyError as e:
        assert 'sensitive_data' in str(e) or e.args[0] == 'sensitive_data'

    # Also test valid formatting
    valid_cell = "Safe: {num_classes}, {steps_str}, {num_filters}"
    writetemplate(test_line, valid_cell)

    with open(test_line, 'r') as f:
        output = f.read()

    assert output == "Safe: 3, 4800,5400, 24"

    if os.path.exists(test_line):
        os.remove(test_line)

def test_yolov5_writetemplate_valid():
    """
    Tests that yolov5 writetemplate correctly formats the intended variables.
    """
    func_code = extract_writetemplate('yolov5/deeptrash_yolov5.py')

    test_env = {}
    exec(func_code, test_env)
    writetemplate = test_env['writetemplate']

    test_line = 'test_yolov5_valid_output.yaml'
    test_cell = "nc: {num_classes}"

    test_env['num_classes'] = '80'

    writetemplate(test_line, test_cell)

    with open(test_line, 'r') as f:
        output = f.read()

    assert output == "nc: 80"

    if os.path.exists(test_line):
        os.remove(test_line)
