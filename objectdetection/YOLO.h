#pragma once

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <functional>
#include <fstream>
#include <memory>
#include <optional>

enum MODEL_TYPE {
    // FLOAT32 MODEL
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,
    // FLOAT16 MODEL
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
    YOLO_CLS_HALF = 6
};


typedef struct _DL_INIT_PARAM {
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_DETECT_V8;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    int keyPointsNum = 2;
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;


typedef struct _DL_RESULT {
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;


class YOLO {
public:
    YOLO();
    ~YOLO();

public:
    // 模型加载（只需调用一次），失败返回 false 并在 errorMessage 填充信息
    bool loadModel(const DL_INIT_PARAM& params, std::string& errorMessage);

    // 预热（可选）
    bool warmUp();

    // 单张推理
    bool infer(const cv::Mat& imageBgr, std::vector<DL_RESULT>& results);
    // 批量推理（同尺寸预处理，内部拼 batch）
    bool inferBatch(const std::vector<cv::Mat>& imagesBgr, std::vector<std::vector<DL_RESULT>>& batchResults);

    // 类别设置/获取（若模型元数据未包含names可自定义设置）
    void setClasses(const std::vector<std::string>& classNames);
    const std::vector<std::string>& getClasses() const;

    // 提供简单可视化接口
    static cv::Mat drawDetections(const cv::Mat& imageBgr, const std::vector<DL_RESULT>& results,
                                  const std::vector<std::string>& classNames, const std::vector<cv::Scalar>& colors, int thickness = 2);

private:
    struct PreprocessInfo {
        float scale = 1.0f;
        int padX = 0;
        int padY = 0;
        int outW = 0;
        int outH = 0;
    };

    cv::Mat preprocessLetterboxBgrToRgb(const cv::Mat& imageBgr, const std::vector<int>& targetSize,
                                        PreprocessInfo& info) const;
    void blobFromImageCHW01(const cv::Mat& rgb, float* blobPtr) const;
    void postprocessDetections(const float* output, const std::vector<int64_t>& outputShape,
                               const PreprocessInfo& pp, std::vector<DL_RESULT>& results) const;
    void postprocessDetectionsBatch(const float* output, const std::vector<int64_t>& outputShape,
                                    const std::vector<PreprocessInfo>& ppInfos, std::vector<std::vector<DL_RESULT>>& batchResults) const;
    static std::vector<std::string> parsePlaneNames(const char* str);
    static std::string wideErrorToString(const std::exception& e);
    static void buildDistinctColors(size_t numClasses, std::vector<cv::Scalar>& outColors);

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::RunOptions options;
    std::vector<std::string> inputNodeNames;
    std::vector<std::string> outputNodeNames;

    MODEL_TYPE modelType = YOLO_DETECT_V8;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6f;
    float iouThreshold = 0.5f;

    std::vector<std::string> classes;
    std::vector<cv::Scalar> colores;
};
