## 2024-05-24 - Torch Tensor Iteration Penalty
**Learning:** Iterating over YOLOv8 bounding box result objects (`results[].boxes`) and converting properties (`.xyxy[0].tolist()`, `.cls.item()`, `.conf.item()`) one by one in a Python loop incurs massive overhead.
**Action:** Always batch convert Torch tensors to Python lists/arrays FIRST (e.g., `boxes.xyxy.tolist()`, `boxes.cls.tolist()`) before iterating, or use vectorized operations.
