
## 2024-05-28 - Avoid element-by-element Torch tensor iteration overhead in Ultralytics YOLO Results
**Learning:** Extracting values from PyTorch tensors element-by-element within a Python loop (e.g., using `.tolist()` or `.item()` on each box individually) creates significant iteration overhead when parsing Ultralytics YOLO inference results.
**Action:** When parsing `Results.boxes` tensors in loops, batch convert the entire tensors to Python lists first (e.g., `boxes.xyxy.tolist()`, `boxes.cls.tolist()`) and iterate using `zip()` to eliminate this overhead.
