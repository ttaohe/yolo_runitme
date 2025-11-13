# yolo_runtime / yolo_runitme

跨语言的 YOLO ONNX 推理工程：
- C++ 核心（ONNX Runtime + OpenCV），提供 EXE 与 C 接口 DLL；
- pybind11 扩展 `yolort` 与 Python 高层封装 `yolo_runtime.py`（生产者-消费者、滑窗、统计、可视化）。

## 目录结构
```
objectdetection/
  CMakeLists.txt          # Windows 优先（也可适配其它平台）
  YOLO.h / YOLO.cpp       # C++ 核心推理
  ObejectDetect.cpp       # 示例 EXE（单图/批/保存）
  c_api.h / c_api.cpp     # C 接口 DLL（给 C++/Qt 等使用）
  yolo_bindings.cpp       # pybind11 绑定，生成 yolort.pyd
python/
  README.md               # Python 使用说明
  yolo_runtime.py         # Python 运行时（队列/滑窗/统计/绘图）
  demo.py                 # 示例（含大图滑窗模拟）
```

## 依赖
- ONNX Runtime（Windows x64，推荐 1.19.x）
- OpenCV（Windows x64，示例使用 4.5.4 world）
- Python 3.12 x64（如需 Python 侧）

三方库可放在 `objectdetection/3rdPart/` 下，或通过 CMake 变量覆盖。

## Windows 构建（CMake + MSVC）
推荐在“x64 Native Tools Command Prompt for VS 2022”中：
```bat
cd objectdetection
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DOpenCV_DIR="F:/path/to/opencv/build" ^
  -DONNXRUNTIME_ROOT="F:/path/to/onnxruntime-win-x64-1.19.0" ^
  -DONNXRUNTIME_DLL="F:/path/to/onnxruntime-win-x64-1.19.0/onnxruntime.dll" ^
  -DBUILD_PYBIND=ON -DUSE_FETCHCONTENT_PYBIND11=ON
cmake --build build --config Release -j
```
产物：
- `build/bin/Release/objectdetect_app.exe`
- `build/bin/Release/yolo_capi.dll`
- `build/bin/Release/yolort.pyd`

注意运行期 DLL：请将 `onnxruntime.dll` 与 `opencv_world*.dll` 放在可执行/模块同目录或 PATH 中。

## Python 使用
详见 `python/README.md`。典型流程：
```python
from yolo_runtime import YoloRuntime
import cv2

engine = YoloRuntime(model_path="model.onnx", img_size=(640,640), conf=0.25, iou=0.5)
engine.submit_path("image.jpg", image_id="img1", sliding=True)
res = engine.get_result("img1", timeout=10.0)
vis = engine.draw(cv2.imread("image.jpg"), res, draw_label=True, draw_conf=False)
cv2.imwrite("res.jpg", vis)
engine.close()
```

## GitHub 推送
初始化（若尚未初始化）：
```bash
git init
git add .
git commit -m "init: C++ core + pybind11 + Python runtime"
git branch -M main
git remote add origin https://github.com/ttaohe/yolo_runitme.git
git push -u origin main
```
提示：仓库名似乎是 `yolo_runitme`（可能拼写为 yolo_runtime），若非有意可在 GitHub 上重命名仓库并更新远程。 


