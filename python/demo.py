from yolo_runtime import YoloRuntime
import cv2
import numpy as np
import time

def _letterbox_top_left(bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
	# 模拟 C++ 端的“左上角 letterbox”几何（保持 BGR 通道）
	h, w = bgr.shape[:2]
	scale_w = w / float(out_w)
	scale_h = h / float(out_h)
	scale = max(scale_w, scale_h) if max(scale_w, scale_h) > 0 else 1.0
	new_w = int(round(w / scale))
	new_h = int(round(h / scale))
	resized = cv2.resize(bgr, (new_w, new_h))
	padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
	padded[:new_h, :new_w] = resized
	return padded
engine = YoloRuntime(
    model_path="model.onnx",
    img_size=(640, 640),
    conf=0.25,
    iou=0.5,
    classes=["small-vehicle","large-vehicle","plane","storage-tank","ship","harbor",
             "ground-track-field","soccer-ball-field","tennis-court","swimming-pool",
             "baseball-diamond","roundabout","basketball-court","bridge","helicopter"],
    batch_size=8,
    overlap=0.2,
    detect_classes=["small-vehicle", "large-vehicle"],
    show_label=False,
    show_conf=False
)

# 仅设置任务类型并提交任务
# 选项: "small_single" | "big_single"
TASK = "big_single"

if TASK == "small_single":
	res, stats = engine.infer_small_image_path("5.png", timeout=10.0)
	print("FPS:", stats.get("fps", 0.0), stats)
	# 可视化
	src = cv2.imread("5.png")
	if src is not None and res:
		vis = engine.draw(src, res, draw_label=False, draw_conf=False)
		cv2.imwrite("res.jpg", vis)
elif TASK == "big_single":
	# 模拟一个 10000x10000 的大图：用 5.png 平铺
	src = cv2.imread("5.png")
	if src is None:
		raise SystemExit("failed to read 5.png")
	th, tw = 2000, 2000
	# 先按网络预处理几何（左上角 letterbox）生成 640x640 单元，再进行平铺
	unit = _letterbox_top_left(src, engine._img_size[0], engine._img_size[1])
	rep_y = (th + unit.shape[0] - 1) // unit.shape[0]
	rep_x = (tw + unit.shape[1] - 1) // unit.shape[1]
	big = np.tile(unit, (rep_y, rep_x, 1))[:th, :tw].copy()
	vis = big.copy()
	# 流式滑窗推理：逐 tile 回调 FPS，最后产出 final
	last_save = 0.0
	for ev in engine.infer_big_image_stream(big, overlap=0.0, timeout_final=999999):
		if ev.get("kind") == "tile":
			engine.draw_on(vis, ev.get("results", []), draw_label=False, draw_conf=False)
			print(f"tile {ev.get('tiles_done',0)}/{ev.get('tiles_total',0)} fps={ev.get('tile_fps',0.0):.2f}")
			# 定期写盘预览，避免频繁I/O（例如每500ms一次）
			now = time.time()
			if now - last_save >= 0.5:
				cv2.imwrite("res.jpg", vis)
				last_save = now
		elif ev.get("kind") == "final":
			engine.draw_on(vis, ev.get("results", []), draw_label=False, draw_conf=False)
			print("FPS:", ev.get("fps", 0.0), ev.get("stats", {}))
			# 生成并保存框的掩码（与大图同尺寸，二值255）
			mask = engine.boxes_to_mask(vis.shape, ev.get("results", []), filled=True, per_class=False)
			cv2.imwrite("mask.png", mask)
	cv2.imwrite("res.jpg", vis)
else:
	raise SystemExit(f"Unknown TASK: {TASK}")
engine.close()