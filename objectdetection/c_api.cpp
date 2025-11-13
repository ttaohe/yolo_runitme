#include "c_api.h"
#include "YOLO.h"
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

struct YOLOWrapper {
	std::unique_ptr<YOLO> yolo;
	YOLOWrapper() : yolo(std::make_unique<YOLO>()) {}
};

extern "C" {

YOLO_API YOLOHandle yolo_create() {
	try {
		auto* h = new YOLOWrapper();
		return reinterpret_cast<YOLOHandle>(h);
	}
	catch (...) {
		return nullptr;
	}
}

YOLO_API void yolo_destroy(YOLOHandle handle) {
	if (!handle) return;
	auto* h = reinterpret_cast<YOLOWrapper*>(handle);
	delete h;
}

YOLO_API int yolo_load_model(YOLOHandle handle,
	const char* model_path_utf8,
	int img_w,
	int img_h,
	float conf,
	float iou,
	int model_type,
	int intra_threads,
	int log_level) {
	if (!handle || !model_path_utf8) return -1;
	auto* h = reinterpret_cast<YOLOWrapper*>(handle);
	DL_INIT_PARAM p;
	p.modelPath = std::string(model_path_utf8);
	p.imgSize = { img_w, img_h };
	p.rectConfidenceThreshold = conf;
	p.iouThreshold = iou;
	p.modelType = static_cast<MODEL_TYPE>(model_type);
	p.intraOpNumThreads = intra_threads;
	p.logSeverityLevel = log_level;
	std::string err;
	return h->yolo->loadModel(p, err) ? 0 : -2;
}

YOLO_API int yolo_set_classes(YOLOHandle handle, const char** names, int count) {
	if (!handle) return -1;
	auto* h = reinterpret_cast<YOLOWrapper*>(handle);
	std::vector<std::string> cls;
	cls.reserve(static_cast<size_t>(count));
	for (int i = 0; i < count; ++i) {
		cls.emplace_back(names[i] ? names[i] : "");
	}
	h->yolo->setClasses(cls);
	return 0;
}

YOLO_API int yolo_warm_up(YOLOHandle handle) {
	if (!handle) return -1;
	auto* h = reinterpret_cast<YOLOWrapper*>(handle);
	return h->yolo->warmUp() ? 0 : -2;
}

static void fill_results(const std::vector<DL_RESULT>& from, CDetResult* out_results, int max_results, int* out_count) {
	int n = static_cast<int>(from.size());
	if (out_count) *out_count = n;
	if (!out_results || max_results <= 0) return;
	int copy_n = std::min(n, max_results);
	for (int i = 0; i < copy_n; ++i) {
		out_results[i].classId = from[i].classId;
		out_results[i].confidence = from[i].confidence;
		out_results[i].x = from[i].box.x;
		out_results[i].y = from[i].box.y;
		out_results[i].w = from[i].box.width;
		out_results[i].h = from[i].box.height;
	}
}

YOLO_API int yolo_infer_image_path(YOLOHandle handle,
	const char* image_path_utf8,
	CDetResult* out_results,
	int max_results,
	int* out_count) {
	if (!handle || !image_path_utf8) return -1;
	auto* h = reinterpret_cast<YOLOWrapper*>(handle);
	cv::Mat img = cv::imread(image_path_utf8);
	if (img.empty()) return -3;
	if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	std::vector<DL_RESULT> results;
	if (!h->yolo->infer(img, results)) return -2;
	fill_results(results, out_results, max_results, out_count);
	return 0;
}

YOLO_API int yolo_infer_image_bgr(YOLOHandle handle,
	const uint8_t* bgr,
	int width,
	int height,
	int stride,
	CDetResult* out_results,
	int max_results,
	int* out_count) {
	if (!handle || !bgr || width <= 0 || height <= 0 || stride <= 0) return -1;
	auto* h = reinterpret_cast<YOLOWrapper*>(handle);
	cv::Mat img(height, width, CV_8UC3, const_cast<uint8_t*>(bgr), static_cast<size_t>(stride));
	std::vector<DL_RESULT> results;
	if (!h->yolo->infer(img, results)) return -2;
	fill_results(results, out_results, max_results, out_count);
	return 0;
}

}


