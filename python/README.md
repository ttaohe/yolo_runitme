# Python 推理引擎使用说明

本目录提供 pybind11 扩展模块 `yolort` 与高层封装 `yolo_runtime.py`，用于在 Python 中进行 YOLO ONNX 模型的高效推理。

## 1. 依赖与环境
- Python 3.12（64 位）
- 运行时 DLL（需要与编译时版本一致）：
  - onnxruntime.dll（建议 1.19.x）
  - opencv_world454.dll（或与你使用的 OpenCV 版本一致的 world DLL）
- 将上述 DLL 放在 `yolort.pyd` 同目录，或加入到 PATH。

## 2. 构建 yolort（可选）
若已由 CMake 构建生成 `yolort.cp312-win_amd64.pyd`，可跳过本节。

使用 CMake（示例）
```bash
cmake -S objectdetection -B build -G "Visual Studio 17 2022" -A x64 ^
  -DOpenCV_DIR=".../opencv/build" ^
  -DONNXRUNTIME_ROOT=".../onnxruntime-win-x64-1.19.0" ^
  -DONNXRUNTIME_DLL=".../onnxruntime-win-x64-1.19.0/onnxruntime.dll" ^
  -DBUILD_PYBIND=ON -DUSE_FETCHCONTENT_PYBIND11=ON
cmake --build build --config Release -j
```
产物 `yolort.pyd` 位于 `build/bin/Release/`。

## 3. API 概览（yolo_runtime.YoloRuntime）
```python
from yolo_runtime import YoloRuntime

engine = YoloRuntime(
    model_path="model.onnx",
    img_size=(640, 640),
    conf=0.25,
    iou=0.5,
    classes=[...],                 # 可选：覆盖模型内置类别名
    batch_size=8,
    max_queue=64,
    overlap=0.2,
    detect_classes=["plane", 4],   # 可选：仅检测指定类（名称或ID）
    show_label=True,               # 可选：绘制时默认显示类别
    show_conf=True                 # 可选：绘制时默认显示置信度
)

# 提交图像（内存）或路径（内部读取）
engine.submit(image_bgr, image_id="img1", sliding=True)
engine.submit_path("img.jpg", image_id="img2", sliding=False, resize_to_input=False)

# 非阻塞获取结果
res = engine.get_result("img1", timeout=5.0)  # None=超时未完成

# 绘制
vis = engine.draw(image_bgr, res, draw_label=True, draw_conf=False)

# 统计
engine.reset_stats()
fps = engine.get_fps()
stats = engine.get_stats()        # {"count":N, "elapsed":T, "fps":...}
tile = engine.get_tile_stats()    # {"tiles_done":..., "tiles_total":..., "elapsed":..., "fps":...}
proc, exp = engine.get_progress("img1")

engine.close()
```

## 4. 滑窗与进度
- `sliding=True` 时，按 `img_size` 对大图进行平铺滑窗；`overlap` 控制重叠比例。
- 可结合 `tqdm` 实时显示进度：参考 `python/demo.py`。

## 5. 设计要点
- 推理引擎使用生产者-消费者模型：一个生产线程拆分任务与滑窗，另一个消费者线程进行（批量）推理与聚合。
- C++ 推理（pybind11）释放 GIL，避免阻塞 Python 主线程，使进度刷新流畅。
- 后处理兼容 YOLOv8 常见输出 `[1,84,S]` 与 `[1,S,84]`。
- FPS 统计忽略模型加载与预热时间；滑窗时提供 tile 级别统计（tile/s）。

## 6. 常见问题
- 导入失败：确认 `yolort.pyd` 与依赖 DLL 在同一目录或在 PATH 中，且 Python/构建/第三方库均为 x64。
- ORT 版本冲突：若提示 API 版本不匹配，确保加载的是与你编译时一致的 onnxruntime.dll（例如 1.19.x）。

## 7. 示例
请参考 `python/demo.py`：包含普通图与 10000x10000 模拟大图的推理、进度与可视化示例。 


