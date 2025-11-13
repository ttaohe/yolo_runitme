from yolo_runtime import YoloRuntime
import cv2
import numpy as np
import time

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
	th, tw = 10000, 10000
	h, w = src.shape[:2]
	rep_y = (th + h - 1) // h
	rep_x = (tw + w - 1) // w
	big = np.tile(src, (rep_y, rep_x, 1))
	big = big[:th, :tw].copy()
	vis = big.copy()
	# 流式滑窗推理：逐 tile 回调 FPS，最后产出 final
	last_save = 0.0
	for ev in engine.infer_big_image_stream(big, timeout_final=999999):
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
	cv2.imwrite("res.jpg", vis)
else:
	raise SystemExit(f"Unknown TASK: {TASK}")
engine.close()