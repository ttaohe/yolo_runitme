import os, sys
import ctypes

from yolo_runtime import YoloRuntime
import cv2
import numpy as np
try:
	from tqdm import tqdm
except Exception:
	tqdm = None

engine = YoloRuntime(
    model_path="model.onnx",
    img_size=(640, 640),
    conf=0.01,
    iou=0.5,
    classes=["small-vehicle","large-vehicle","plane","storage-tank","ship","harbor",
             "ground-track-field","soccer-ball-field","tennis-court","swimming-pool",
             "baseball-diamond","roundabout","basketball-court","bridge","helicopter"],
    batch_size=8,
    overlap=0.2
)

# 模拟超大图 10000x10000：用 5.png 进行平铺拼接，然后裁剪到目标尺寸
SIM_BIG = False
if SIM_BIG:
	src = cv2.imread("5.png")
	if src is None:
		raise SystemExit("failed to read 5.png")
	th, tw = 10000, 10000
	h, w = src.shape[:2]
	rep_y = (th + h - 1) // h
	rep_x = (tw + w - 1) // w
	big = np.tile(src, (rep_y, rep_x, 1))
	big = big[:th, :tw].copy()

	# 滑窗推理（内部按 img_size 分块），可调 overlap；同时显示进度与tile FPS
	engine.reset_stats()
	engine.submit(big, image_id="big1", sliding=True)
	res = None
	pbar = None
	expected = None
	while True:
		# 非阻塞获取结果
		tmp = engine.get_result("big1", timeout=0.1)
		if tmp is not None:
			res = tmp
			if pbar is not None:
				pbar.n = pbar.total
				pbar.refresh()
			break
		# 进度与FPS刷新
		proc, exp = engine.get_progress("big1")
		if exp and expected is None and tqdm is not None:
			expected = exp
			pbar = tqdm(total=exp, desc="Sliding inference", unit="tile")
		if pbar is not None:
			pbar.n = proc
			stats = engine.get_tile_stats()
			pbar.set_postfix_str(f"tile_fps={stats['fps']:.2f}")
			pbar.refresh()
	print("Image FPS:", engine.get_fps(), engine.get_stats())

	# 可视化与保存
	vis = engine.draw(big, res, draw_label=True, draw_conf=False)
	cv2.imwrite("res.jpg", vis)
else:
	engine.submit_path("5.png", image_id="img1", sliding=False, resize_to_input=False)
	res = engine.get_result("img1", timeout=10.0)
	print("FPS:", engine.get_fps(), engine.get_stats())
	vis = cv2.imread("5.png")
	if vis is None:
		raise SystemExit("failed to read 5.png for drawing")
	if res:
		for det in res:
			b = det.get("box", {})
			x, y, w, h = int(b.get("x", 0)), int(b.get("y", 0)), int(b.get("w", 0)), int(b.get("h", 0))
			cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
			label = f'{det.get("class_id", 0)} {det.get("confidence", 0.0):.2f}'
			(tw2, th2), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
			cv2.rectangle(vis, (x, max(0, y - th2 - 6)), (x + tw2 + 6, y), (0, 0, 255), -1)
			cv2.putText(vis, label, (x + 3, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
	cv2.imwrite("res.jpg", vis)
engine.close()