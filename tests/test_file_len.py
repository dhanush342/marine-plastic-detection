import ast
import tempfile
import os

def get_file_len_function():
    with open('yolov4/deeptrash_yolov4.py', 'r') as f:
        source = f.read()

    # Clean the source to remove invalid IPython syntax
    lines = source.split('\n')
    clean_lines = []
    for line in lines:
        if '!' in line and 'gpu_info =' in line:
            clean_lines.append('gpu_info = ""')
        elif line.lstrip().startswith('!') or line.lstrip().startswith('%'):
            clean_lines.append('pass')
        elif '```' in line:
            clean_lines.append('pass')
        else:
            clean_lines.append(line)

    # There is also javascript syntax in the file after line 464
    # We will just parse up to the javascript part.
    clean_source = '\n'.join(clean_lines)

    # We can truncate it right before the javascript block
    js_start = clean_source.find('function ClickConnect')
    if js_start != -1:
        clean_source = clean_source[:js_start]

    tree = ast.parse(clean_source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'file_len':
            mod = ast.Module(body=[node], type_ignores=[])
            code = compile(mod, filename='<ast>', mode='exec')
            namespace = {}
            exec(code, namespace)
            return namespace['file_len']

    raise ValueError("Could not find file_len function in yolov4/deeptrash_yolov4.py")

file_len = get_file_len_function()

def test_file_len_empty():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_name = f.name

    try:
        assert file_len(temp_name) == 0
    finally:
        os.remove(temp_name)

def test_file_len_multiple_lines():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("line 1\nline 2\nline 3\n")
        temp_name = f.name

    try:
        assert file_len(temp_name) == 3
    finally:
        os.remove(temp_name)

def test_file_len_one_line_no_newline():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("just one line")
        temp_name = f.name

    try:
        assert file_len(temp_name) == 1
    finally:
        os.remove(temp_name)

def test_file_len_blank_lines():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("\n\n\n")
        temp_name = f.name

    try:
        assert file_len(temp_name) == 3
    finally:
        os.remove(temp_name)
