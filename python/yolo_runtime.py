import threading
import queue
import time
from typing import List, Tuple, Optional, Dict, Any, Union, Set
import numpy as np

try:
	import yolort  # pybind11 module
except Exception as e:
	yolort = None


# -----------------------------
# 工具函数：NMS 与滑窗
# -----------------------------
def _nms_iou(box_a, box_b) -> float:
	xa1, ya1, wa, ha = box_a
	xb1, yb1, wb, hb = box_b
	xa2, ya2 = xa1 + wa, ya1 + ha
	xb2, yb2 = xb1 + wb, yb1 + hb
	inter_x1 = max(xa1, xb1)
	inter_y1 = max(ya1, yb1)
	inter_x2 = min(xa2, xb2)
	inter_y2 = min(ya2, yb2)
	inter_w = max(0, inter_x2 - inter_x1)
	inter_h = max(0, inter_y2 - inter_y1)
	inter = inter_w * inter_h
	union = wa * ha + wb * hb - inter
	if union <= 0:
		return 0.0
	return inter / union


def _nms(results: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
	if not results:
		return results
	results = sorted(results, key=lambda d: d["confidence"], reverse=True)
	keep = []
	suppressed = [False] * len(results)
	for i in range(len(results)):
		if suppressed[i]:
			continue
		keep.append(results[i])
		ax = results[i]["box"]["x"]
		ay = results[i]["box"]["y"]
		aw = results[i]["box"]["w"]
		ah = results[i]["box"]["h"]
		for j in range(i + 1, len(results)):
			if suppressed[j]:
				continue
			if results[i]["class_id"] != results[j]["class_id"]:
				continue
			bx = results[j]["box"]["x"]
			by = results[j]["box"]["y"]
			bw = results[j]["box"]["w"]
			bh = results[j]["box"]["h"]
			if _nms_iou((ax, ay, aw, ah), (bx, by, bw, bh)) > iou_thresh:
				suppressed[j] = True
	return keep


def _slide_windows(image: np.ndarray, tile_hw: Tuple[int, int], overlap: float) -> List[Tuple[np.ndarray, int, int]]:
	h, w = image.shape[:2]
	th, tw = tile_hw
	if th <= 0 or tw <= 0:
		raise ValueError("Invalid tile size")
	step_h = max(1, int(th * (1 - overlap)))
	step_w = max(1, int(tw * (1 - overlap)))
	tiles = []
	for y in range(0, max(1, h - th + 1), step_h):
		for x in range(0, max(1, w - tw + 1), step_w):
			tile = image[y:y + th, x:x + tw]
			if tile.shape[0] != th or tile.shape[1] != tw:
				padded = np.zeros((th, tw, 3), dtype=np.uint8)
				padded[:tile.shape[0], :tile.shape[1], :] = tile
				tile = padded
			tiles.append((tile, x, y))
	# 右下角补充
	if h > th:
		y = h - th
		for x in range(0, max(1, w - tw + 1), step_w):
			tile = image[y:y + th, x:x + tw]
			if tile.shape[0] != th or tile.shape[1] != tw:
				padded = np.zeros((th, tw, 3), dtype=np.uint8)
				padded[:tile.shape[0], :tile.shape[1], :] = tile
				tile = padded
			tiles.append((tile, x, y))
	if w > tw:
		x = w - tw
		for y in range(0, max(1, h - th + 1), step_h):
			tile = image[y:y + th, x:x + tw]
			if tile.shape[0] != th or tile.shape[1] != tw:
				padded = np.zeros((th, tw, 3), dtype=np.uint8)
				padded[:tile.shape[0], :tile.shape[1], :] = tile
				tile = padded
			tiles.append((tile, x, y))
	if h > th and w > tw:
		y = h - th
		x = w - tw
		tile = image[y:y + th, x:x + tw]
		if tile.shape[0] != th or tile.shape[1] != tw:
			padded = np.zeros((th, tw, 3), dtype=np.uint8)
			padded[:tile.shape[0], :tile.shape[1], :] = tile
			tile = padded
		tiles.append((tile, x, y))
	return tiles


# -----------------------------
# 推理引擎：清晰职责划分
# -----------------------------
class YoloRuntime:
	def __init__(self,
	             model_path: str,
	             img_size: Tuple[int, int] = (640, 640),
	             conf: float = 0.25,
	             iou: float = 0.5,
	             classes: Optional[List[str]] = None,
	             batch_size: int = 8,
	             max_queue: int = 64,
	             overlap: float = 0.2,
	             detect_classes: Optional[List[Union[int, str]]] = None,
	             show_label: bool = True,
	             show_conf: bool = True):
		if yolort is None:
			raise RuntimeError("yolort(pybind11) module not found or failed to load")
		self._engine = yolort.PyYolo()
		ok = self._engine.load_model(model_path, img_size[1], img_size[0], conf, iou, 1, 1, 3)
		if not ok:
			raise RuntimeError("load_model failed")
		if classes:
			self._engine.set_classes(classes)
		self._engine.warm_up()

		# 基础配置
		self._img_size = img_size
		self._overlap = overlap
		self._batch_size = batch_size
		self._show_label = show_label
		self._show_conf = show_conf

		# 队列与状态
		self._images_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(max_queue)
		self._tiles_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(max_queue * 4)
		self._results_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(max_queue)
		self._expected_tiles: Dict[str, int] = {}
		self._processed_tiles: Dict[str, int] = {}
		self._acc_results: Dict[str, List[Dict[str, Any]]] = {}
		self._lock = threading.Lock()

		# 类别过滤
		self._allowed_class_ids: Optional[Set[int]] = self._resolve_allowed_ids(detect_classes)

		# 后台线程
		self._stopping = threading.Event()
		self._producer = threading.Thread(target=self._producer_loop, daemon=True)
		self._consumer = threading.Thread(target=self._consumer_loop, daemon=True)
		self._producer.start()
		self._consumer.start()

		# 统计
		self._t0: Optional[float] = None
		self._num_done: int = 0
		self._t0_tiles: Optional[float] = None
		self._tiles_total: int = 0
		self._tiles_done: int = 0

	# ---------- 公共API ----------
	def close(self):
		self._stopping.set()
		for q in (self._images_q, self._tiles_q):
			try:
				q.put_nowait({"stop": True})
			except Exception:
				pass
		self._producer.join(timeout=5.0)
		self._consumer.join(timeout=5.0)
		self._engine = None

	def submit(self, image_bgr: np.ndarray, image_id: str, sliding: bool = True,
	           tile_size: Optional[Tuple[int, int]] = None, overlap: Optional[float] = None):
		if tile_size is None:
			tile_size = self._img_size
		if overlap is None:
			overlap = self._overlap
		self._images_q.put({
			"id": image_id,
			"image": image_bgr,
			"sliding": sliding,
			"tile_size": tile_size,
			"overlap": overlap
		})

	def submit_path(self, image_path: str, image_id: str, sliding: bool = True,
	                tile_size: Optional[Tuple[int, int]] = None, overlap: Optional[float] = None,
	                resize_to_input: bool = False) -> None:
		import cv2
		img = cv2.imread(image_path)
		if img is None:
			raise RuntimeError(f"Failed to read image: {image_path}")
		if resize_to_input:
			img = cv2.resize(img, (self._img_size[0], self._img_size[1]), interpolation=cv2.INTER_LINEAR)
		self.submit(img, image_id=image_id, sliding=sliding, tile_size=tile_size, overlap=overlap)

	def get_result(self, image_id: str, timeout: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
		deadline = None if timeout is None else (time.time() + timeout)
		while True:
			remain = None if deadline is None else max(0.0, deadline - time.time())
			try:
				item = self._results_q.get(timeout=remain)
			except queue.Empty:
				return None
			if item.get("id") == image_id:
				return item.get("results", [])
			else:
				self._results_q.put(item)
				if deadline is not None and time.time() >= deadline:
					return None

	def draw(self, image_bgr: np.ndarray, results: List[Dict[str, Any]],
	         color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2,
	         draw_label: Optional[bool] = None, draw_conf: Optional[bool] = None) -> np.ndarray:
		if draw_label is None:
			draw_label = getattr(self, "_show_label", True)
		if draw_conf is None:
			draw_conf = getattr(self, "_show_conf", True)
		try:
			classes = self._engine.get_classes()
		except Exception:
			classes = []
		import cv2
		vis = image_bgr.copy()
		for det in results or []:
			b = det.get("box", {})
			x, y, w, h = int(b.get("x", 0)), int(b.get("y", 0)), int(b.get("w", 0)), int(b.get("h", 0))
			cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
			label_parts = []
			if draw_label:
				cid = int(det.get("class_id", 0))
				name = classes[cid] if (cid >= 0 and cid < len(classes)) else str(cid)
				label_parts.append(str(name))
			if draw_conf:
				label_parts.append(f'{float(det.get("confidence", 0.0)):.2f}')
			if label_parts:
				label = " ".join(label_parts)
				(tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
				cv2.rectangle(vis, (x, max(0, y - th - 6)), (x + tw + 6, y), color, -1)
				cv2.putText(vis, label, (x + 3, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
		return vis

	# 统计
	def get_fps(self) -> float:
		if self._t0 is None or self._num_done <= 0:
			return 0.0
		elapsed = max(1e-6, time.time() - self._t0)
		return float(self._num_done / elapsed)

	def get_stats(self) -> Dict[str, Any]:
		if self._t0 is None:
			return {"count": 0, "elapsed": 0.0, "fps": 0.0}
		elapsed = max(0.0, time.time() - self._t0)
		fps = float(self._num_done / max(1e-6, elapsed))
		return {"count": int(self._num_done), "elapsed": float(elapsed), "fps": fps}

	def reset_stats(self):
		self._t0 = None
		self._num_done = 0
		self._t0_tiles = None
		self._tiles_total = 0
		self._tiles_done = 0

	def get_progress(self, image_id: str) -> Tuple[int, int]:
		with self._lock:
			p = self._processed_tiles.get(image_id, 0)
			e = self._expected_tiles.get(image_id, 0)
			return int(p), int(e)

	def get_tile_stats(self) -> Dict[str, Any]:
		with self._lock:
			done = int(self._tiles_done)
			total = int(self._tiles_total)
			t0 = self._t0_tiles
		if t0 is None:
			return {"tiles_done": done, "tiles_total": total, "elapsed": 0.0, "fps": 0.0}
		elapsed = max(1e-6, time.time() - t0)
		return {"tiles_done": done, "tiles_total": total, "elapsed": float(elapsed), "fps": float(done / elapsed)}

	# ---------- 内部方法 ----------
	def _resolve_allowed_ids(self, detect_classes: Optional[List[Union[int, str]]]) -> Optional[Set[int]]:
		if detect_classes is None:
			return None
		allowed: Set[int] = set()
		try:
			cls_list = self._engine.get_classes()
		except Exception:
			cls_list = []
		for it in detect_classes:
			if isinstance(it, int):
				allowed.add(int(it))
			else:
				if it in cls_list:
					allowed.add(int(cls_list.index(it)))
		return allowed if allowed else None

	def _producer_loop(self):
		while not self._stopping.is_set():
			try:
				req = self._images_q.get(timeout=0.1)
			except queue.Empty:
				continue
			if req.get("stop"):
				break
			image_id = req["id"]
			image = req["image"]
			sliding = req["sliding"]
			tile_hw = (req["tile_size"][1], req["tile_size"][0])  # (h,w)
			overlap = req["overlap"]
			if not sliding:
				with self._lock:
					self._expected_tiles[image_id] = 1
					self._processed_tiles[image_id] = 0
					self._acc_results[image_id] = []
					self._tiles_total += 1
				self._tiles_q.put({"id": image_id, "tile": image, "ox": 0, "oy": 0, "shape": image.shape})
			else:
				tiles = _slide_windows(image, tile_hw, overlap)
				with self._lock:
					self._expected_tiles[image_id] = len(tiles)
					self._processed_tiles[image_id] = 0
					self._acc_results[image_id] = []
					self._tiles_total += len(tiles)
				for tile, ox, oy in tiles:
					self._tiles_q.put({"id": image_id, "tile": tile, "ox": ox, "oy": oy, "shape": image.shape})

	def _consumer_loop(self):
		while not self._stopping.is_set():
			batch = []
			meta = []
			try:
				item = self._tiles_q.get(timeout=0.05)
			except queue.Empty:
				continue
			if item.get("stop"):
				break
			batch.append(item["tile"])
			meta.append(item)
			try:
				while len(batch) < self._batch_size:
					item = self._tiles_q.get_nowait()
					if item.get("stop"):
						self._stopping.set()
						break
					batch.append(item["tile"])
					meta.append(item)
			except queue.Empty:
				pass
			# 推理：优先 batch，失败回退逐 tile
			# 第一次真正开始推理时，启动整体计时器（忽略加载/预热/排队时间）
			if self._t0 is None:
				self._t0 = time.time()
			try:
				results_batch = self._engine.infer_batch(batch) if len(batch) > 1 else [self._engine.infer(batch[0])]
			except Exception:
				results_batch = []
				for t in batch:
					try:
						results_batch.append(self._engine.infer(t))
					except Exception:
						results_batch.append([])
			for res_list, m in zip(results_batch, meta):
				ox, oy = m["ox"], m["oy"]
				image_id = m["id"]
				shifted = []
				for r in res_list:
					if self._allowed_class_ids is not None and int(r["class_id"]) not in self._allowed_class_ids:
						continue
					b = r["box"].copy()
					b["x"] += ox
					b["y"] += oy
					shifted.append({"class_id": int(r["class_id"]), "confidence": float(r["confidence"]), "box": b})
				with self._lock:
					self._acc_results[image_id].extend(shifted)
					self._tiles_done += 1
					if self._t0_tiles is None:
						self._t0_tiles = time.time()
					self._processed_tiles[image_id] += 1
					if self._processed_tiles[image_id] >= self._expected_tiles[image_id]:
						final = _nms(self._acc_results[image_id], iou_thresh=0.5)
						self._results_q.put({"id": image_id, "results": final})
						self._num_done += 1
						del self._acc_results[image_id]
						del self._processed_tiles[image_id]
						del self._expected_tiles[image_id]

