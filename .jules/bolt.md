
## 2025-02-28 - Optimizing Ultralytics YOLO Results Parsing
**Learning:** Element-by-element `.tolist()` or `.item()` calls on `Results.boxes` tensors inside loops cause significant Torch tensor iteration overhead, which severely bottlenecks API prediction parsing on multiple bounding boxes.
**Action:** When extracting multiple attributes from Ultralytics YOLO `Results.boxes` tensors, batch convert the entire tensors to Python lists first (e.g., `boxes.xyxy.tolist()`, `boxes.cls.tolist()`) and iterate using `zip()` to eliminate the overhead.
