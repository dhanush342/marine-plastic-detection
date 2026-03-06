import os

class MockFiles:
    def upload(self):
        return {
            "../../../etc/passwd": b"fake passwd data",
            "normal_image.jpg": b"normal image data",
            "some/nested/path/to/file.png": b"nested file data"
        }

def mock_upload():
    uploaded = MockFiles().upload()
    results = []
    for name, data in uploaded.items():
        secure_name = os.path.basename(name)
        results.append(secure_name)
    return results

if __name__ == "__main__":
    results = mock_upload()
    expected = ["passwd", "normal_image.jpg", "file.png"]
    assert results == expected, f"Expected {expected}, got {results}"
    print("Test passed: Path traversal payloads were successfully sanitized!")
