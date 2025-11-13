    cd /d F:\02-nm\hjj-2\windows-x64\ObjectDetection\objectdetection\objectdetection
    rmdir /s /q build
    cmake -S . -B build -G "Ninja" -DOpenCV_DIR="F:/02-nm/hjj-2/windows-x64/ObjectDetection/objectdetection/3rdPart/opencv454-win/opencv/build" -DONNXRUNTIME_ROOT="F:/02-nm/hjj-2/windows-x64/ObjectDetection/objectdetection/3rdPart/onnxruntime-win-x64-1.19.0" -DONNXRUNTIME_DLL="F:/02-nm/hjj-2/windows-x64/ObjectDetection/objectdetection/3rdPart/onnxruntime-win-x64-1.19.0/onnxruntime.dll" -DBUILD_PYBIND=ON -DUSE_FETCHCONTENT_PYBIND11=ON
    cmake --build build --config Release -j
    ```
- 构建成功后，yolort.pyd 在 build/bin/Release/
  - 确保该目录在 Python 的 sys.path（可把 pyd 拷到你的项目，或设置 PYTHONPATH）。

Python 侧快速测试（基于我们写好的生产者-消费者类）
```python
from yolo_runtime import YoloRuntime
import cv2

engine = YoloRuntime(
    model_path="F:/.../model.onnx",
    img_size=(640, 640),
    conf=0.25,
    iou=0.5,
    classes=["small-vehicle","large-vehicle","plane","storage-tank","ship","harbor",
             "ground-track-field","soccer-ball-field","tennis-court","swimming-pool",
             "baseball-diamond","roundabout","basketball-court","bridge","helicopter"],
    batch_size=8,
    overlap=0.2
)

img = cv2.imread("F:/.../test.jpg")
engine.submit(img, image_id="img1", sliding=True)
res = engine.get_result("img1", timeout=10.0)
print(res)
engine.close()
```

常见注意事项
- Python 必须是 64 位，与 MSVC/构建目标一致。
- 运行时 DLL：
  - 我们已自动拷贝 onnxruntime.dll 到目标输出目录。
  - OpenCV 的 DLL 需可被找到（在 PATH 或与 yolort.pyd 同目录）。若缺失，请将 OpenCV bin 目录（与对应 vcXX/Release 的 .dll）加入 PATH，或拷贝到 build/bin/Release。

如果你想先用 C 接口 yolo_capi.dll 在 Python 里用 ctypes 试跑，也可以；但你已有 yolo_runtime.py（依赖 yolort.pyd），直接构建 yolort 对接会更顺畅。