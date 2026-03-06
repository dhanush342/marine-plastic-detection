## 2024-03-06 - Replacing os.listdir with os.scandir for performance
**Learning:** `os.listdir` creates an entire list of strings in memory, which can be inefficient for large datasets like image directories. `os.scandir` returns an iterator of `DirEntry` objects, which is more memory efficient and faster for directory traversal, especially when filtering.
**Action:** Replace `os.listdir` with `os.scandir` in `yolov4/deeptrash_yolov4.py` and potentially in the Jupyter notebook `yolov4/DeepTrash_Yolov4.ipynb` to improve performance.
