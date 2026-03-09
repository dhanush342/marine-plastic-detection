## 2024-05-24 - PyTorch Tensor Iteration Overhead
**Learning:** Iterating over Torch tensors element-by-element inside Python loops causes significant overhead. Calling `.tolist()` or `.item()` on individual tensor elements within a loop is a performance bottleneck.
**Action:** When working with tensor collections (like Ultralytics YOLO Results.boxes), batch convert the entire tensors to Python lists first (e.g., `boxes.xyxy.tolist()`), and then iterate over them using `zip()`. This eliminates the Torch tensor iteration overhead.
