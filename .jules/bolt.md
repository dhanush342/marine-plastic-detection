## 2026-03-14 - Optimize Ultralytics YOLO Results Parsing
**Learning:** Element-by-element `.tolist()` or `.item()` calls on `Results.boxes` tensors inside loops cause significant Torch tensor iteration overhead, creating a performance bottleneck during inference result processing.
**Action:** When parsing Ultralytics YOLO predictions, batch convert the entire tensors to Python lists first (e.g., `boxes.xyxy.tolist()`, `boxes.cls.tolist()`, `boxes.conf.tolist()`) and iterate using `zip()` to eliminate the tensor iteration overhead.
