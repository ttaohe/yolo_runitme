#pragma once

#ifdef _WIN32
#define YOLO_API __declspec(dllexport)
#else
#define YOLO_API
#endif

#include <cstdint>

extern "C" {

typedef void* YOLOHandle;

typedef struct CDetResult {
	int classId;
	float confidence;
	int x;
	int y;
	int w;
	int h;
} CDetResult;

YOLO_API YOLOHandle yolo_create();
YOLO_API void yolo_destroy(YOLOHandle handle);

YOLO_API int yolo_load_model(YOLOHandle handle,
	const char* model_path_utf8,
	int img_w,
	int img_h,
	float conf,
	float iou,
	int model_type,
	int intra_threads,
	int log_level);

YOLO_API int yolo_set_classes(YOLOHandle handle, const char** names, int count);
YOLO_API int yolo_warm_up(YOLOHandle handle);

YOLO_API int yolo_infer_image_path(YOLOHandle handle,
	const char* image_path_utf8,
	CDetResult* out_results,
	int max_results,
	int* out_count);

YOLO_API int yolo_infer_image_bgr(YOLOHandle handle,
	const uint8_t* bgr,
	int width,
	int height,
	int stride,
	CDetResult* out_results,
	int max_results,
	int* out_count);

}


