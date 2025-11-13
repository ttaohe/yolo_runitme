import threading
import queue
import time
from typing import List, Tuple, Optional, Dict, Any, Union, Set
import numpy as np
import os

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
		cpu_threads = max(1, (os.cpu_count() or 1))
		ok = self._engine.load_model(model_path, img_size[1], img_size[0], conf, iou, 1, cpu_threads, 3)
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
		self._tile_results_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(max_queue * 8)
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

	# 简化接口：小图/大图/批量任务
	def submit_small_image_path(self, image_path: str, image_id: str) -> None:
		self.submit_path(image_path, image_id=image_id, sliding=False, resize_to_input=False)

	def submit_big_image_path(self, image_path: str, image_id: str, tile_size: Optional[Tuple[int, int]] = None,
	                          overlap: Optional[float] = None) -> None:
		self.submit_path(image_path, image_id=image_id, sliding=True, tile_size=tile_size, overlap=overlap, resize_to_input=False)

	# 一次性小图推理（同步），返回 (results, stats)
	def infer_small_image_path(self, image_path: str, timeout: Optional[float] = 10.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
		self.reset_stats()
		image_id = "small"
		self.submit_small_image_path(image_path, image_id=image_id)
		res = self.get_result(image_id, timeout=timeout) or []
		return res, self.get_stats()

	# 大图滑窗流式推理（同步生成器）：逐 tile 产出 {'kind':'tile', 'ox','oy','results','tile_fps',...}，结束产出 {'kind':'final', 'results','fps','stats'}
	def infer_big_image_stream_path(self, image_path: str,
	                                tile_size: Optional[Tuple[int, int]] = None,
	                                overlap: Optional[float] = None,
	                                timeout_final: Optional[float] = 60.0):
		import cv2
		img = cv2.imread(image_path)
		if img is None:
			raise RuntimeError(f"Failed to read image: {image_path}")
		yield from self.infer_big_image_stream(img, tile_size=tile_size, overlap=overlap, timeout_final=timeout_final)

	def infer_big_image_stream(self, image_bgr: np.ndarray,
	                           tile_size: Optional[Tuple[int, int]] = None,
	                           overlap: Optional[float] = None,
	                           timeout_final: Optional[float] = 60.0):
		self.reset_stats()
		image_id = "big"
		self.submit(image_bgr, image_id=image_id, sliding=True, tile_size=tile_size, overlap=overlap)
		deadline = None if timeout_final is None else (time.time() + timeout_final)
		while True:
			# 产出 tile
			try:
				item = self._tile_results_q.get(timeout=0.1)
			except queue.Empty:
				item = None
			if item is not None and item.get("id") == image_id:
				tstats = self.get_tile_stats()
				yield {
					"kind": "tile",
					"ox": int(item.get("ox", 0)),
					"oy": int(item.get("oy", 0)),
					"results": item.get("results", []),
					"tiles_done": tstats.get("tiles_done", 0),
					"tiles_total": tstats.get("tiles_total", 0),
					"tile_fps": tstats.get("fps", 0.0),
				}
			# 检查最终结果
			final = self.get_result(image_id, timeout=0.0)
			if final is not None:
				yield {"kind": "final", "results": final, "fps": self.get_fps(), "stats": self.get_stats()}
				break
			if deadline is not None and time.time() >= deadline:
				break

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

	def draw_on(self, image_bgr: np.ndarray, results: List[Dict[str, Any]],
	            color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2,
	            draw_label: Optional[bool] = None, draw_conf: Optional[bool] = None) -> np.ndarray:
		"""
		在给定图像上原地绘制，返回同一图像对象（便于大图流式可视化，避免复制）。
		"""
		if draw_label is None:
			draw_label = getattr(self, "_show_label", True)
		if draw_conf is None:
			draw_conf = getattr(self, "_show_conf", True)
		try:
			classes = self._engine.get_classes()
		except Exception:
			classes = []
		import cv2
		for det in results or []:
			b = det.get("box", {})
			x, y, w, h = int(b.get("x", 0)), int(b.get("y", 0)), int(b.get("w", 0)), int(b.get("h", 0))
			cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, thickness)
			if draw_label or draw_conf:
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
					cv2.rectangle(image_bgr, (x, max(0, y - th - 6)), (x + tw + 6, y), color, -1)
					cv2.putText(image_bgr, label, (x + 3, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
		return image_bgr

	def boxes_to_mask(self, image_shape: Tuple[int, int, int], results: List[Dict[str, Any]],
	                  filled: bool = True, per_class: bool = False) -> np.ndarray:
		"""
		根据检测框生成单通道掩码。
		- filled=True 生成实心矩形；否则只绘制轮廓
		- per_class=True 时，像素值使用 (class_id+1) 编码；否则使用 255 二值掩码
		返回: HxW uint8
		"""
		import cv2
		h = int(image_shape[0])
		w = int(image_shape[1])
		mask = np.zeros((h, w), dtype=np.uint8)
		thickness = -1 if filled else 1
		for det in results or []:
			b = det.get("box", {})
			x, y, bw, bh = int(b.get("x", 0)), int(b.get("y", 0)), int(b.get("w", 0)), int(b.get("h", 0))
			if bw <= 0 or bh <= 0:
				continue
			x1 = max(0, x)
			y1 = max(0, y)
			x2 = min(w - 1, x + bw - 1)
			y2 = min(h - 1, y + bh - 1)
			if x2 < x1 or y2 < y1:
				continue
			if per_class:
				val = int(det.get("class_id", 0)) + 1
				val = 255 if val > 255 else val
			else:
				val = 255
			cv2.rectangle(mask, (x1, y1), (x2, y2), int(val), thickness)
		return mask

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
				# 推送流式 tile 结果
				try:
					self._tile_results_q.put_nowait({"id": image_id, "ox": ox, "oy": oy, "results": shifted})
				except Exception:
					pass
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

