#include "YOLO.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <regex>
#include <opencv2/dnn.hpp>

YOLO::YOLO() {
}

YOLO::~YOLO() {
}

std::vector<std::string> YOLO::parsePlaneNames(const char* str) {
	std::vector<std::string> names;
	if (str == nullptr) return names;
	std::string s(str);
	std::string cur;
	bool inQuotes = false;
	char quoteChar = '\0';
	for (char ch : s) {
		if (!inQuotes && (ch == '\"' || ch == '\'')) {
			inQuotes = true;
			quoteChar = ch;
			cur.clear();
		}
		else if (inQuotes && ch == quoteChar) {
			inQuotes = false;
			if (!cur.empty()) {
				names.push_back(cur);
				cur.clear();
			}
		}
		else if (inQuotes) {
			cur.push_back(ch);
		}
	}
	return names;
}

void YOLO::buildDistinctColors(size_t numClasses, std::vector<cv::Scalar>& outColors) {
	outColors.clear();
	if (numClasses == 0) return;
	for (size_t i = 0; i < numClasses; ++i) {
		float hue = static_cast<float>(i) / static_cast<float>(numClasses);
		cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(static_cast<unsigned char>(hue * 179.f), 255, 255));
		cv::Mat bgr;
		cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
		cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
		outColors.emplace_back(c[0], c[1], c[2]);
	}
}

bool YOLO::loadModel(const DL_INIT_PARAM& params, std::string& errorMessage) {
	try {
		rectConfidenceThreshold = params.rectConfidenceThreshold;
		iouThreshold = params.iouThreshold;
		imgSize = params.imgSize;
		modelType = params.modelType;

		env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
		Ort::SessionOptions sessionOption;
		sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		sessionOption.SetIntraOpNumThreads(params.intraOpNumThreads);
		// 让 ORT 在算子内部与算子之间都能利用多核
		sessionOption.SetInterOpNumThreads(params.intraOpNumThreads);
		sessionOption.SetLogSeverityLevel(params.logSeverityLevel);

#ifdef _WIN32
		int wideLength = MultiByteToWideChar(CP_UTF8, 0, params.modelPath.c_str(), static_cast<int>(params.modelPath.length()), nullptr, 0);
		std::wstring wpath(wideLength, L'\0');
		MultiByteToWideChar(CP_UTF8, 0, params.modelPath.c_str(), static_cast<int>(params.modelPath.length()), wpath.data(), wideLength);
		session = std::make_unique<Ort::Session>(env, wpath.c_str(), sessionOption);
#else
		session = std::make_unique<Ort::Session>(env, params.modelPath.c_str(), sessionOption);
#endif

		Ort::AllocatorWithDefaultOptions allocator;
		inputNodeNames.clear();
		outputNodeNames.clear();
		size_t inputNodesNum = session->GetInputCount();
		for (size_t i = 0; i < inputNodesNum; ++i) {
			Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
			inputNodeNames.emplace_back(input_node_name.get());
		}
		size_t outputNodesNum = session->GetOutputCount();
		for (size_t i = 0; i < outputNodesNum; ++i) {
			Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
			outputNodeNames.emplace_back(output_node_name.get());
		}
		options = Ort::RunOptions{ nullptr };

		try {
			auto name = session->GetModelMetadata().LookupCustomMetadataMapAllocated("names", allocator);
			const char* labelname = name.get();
			auto parsed = parsePlaneNames(labelname);
			if (!parsed.empty()) {
				classes = parsed;
			}
		}
		catch (...) {
		}
		if (!classes.empty()) {
			buildDistinctColors(classes.size(), colores);
		}
		return true;
	}
	catch (const std::exception& e) {
		errorMessage = e.what();
		return false;
	}
}

bool YOLO::warmUp() {
	if (!session) return false;
	cv::Mat dummy(imgSize.at(1), imgSize.at(0), CV_8UC3, cv::Scalar(0, 0, 0));
	PreprocessInfo pp;
	cv::Mat input = preprocessLetterboxBgrToRgb(dummy, imgSize, pp);
	std::vector<float> blob(static_cast<size_t>(3 * input.rows * input.cols));
	blobFromImageCHW01(input, blob.data());
	std::vector<int64_t> inputDims = { 1, 3, input.rows, input.cols };
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		blob.data(), blob.size(), inputDims.data(), inputDims.size());
	std::vector<const char*> inNames, outNames;
	for (auto& n : inputNodeNames) inNames.push_back(n.c_str());
	for (auto& n : outputNodeNames) outNames.push_back(n.c_str());
	auto outputTensors = session->Run(options, inNames.data(), &inputTensor, 1, outNames.data(), static_cast<size_t>(outNames.size()));
	return !outputTensors.empty();
}

void YOLO::setClasses(const std::vector<std::string>& classNames) {
	classes = classNames;
	buildDistinctColors(classes.size(), colores);
}

const std::vector<std::string>& YOLO::getClasses() const {
	return classes;
}

cv::Mat YOLO::preprocessLetterboxBgrToRgb(const cv::Mat& imageBgr, const std::vector<int>& targetSize,
	PreprocessInfo& info) const {
	cv::Mat src = imageBgr;
	if (src.channels() == 1) {
		cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
	}
	cv::Mat rgb;
	cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);

	int outW = targetSize.at(0);
	int outH = targetSize.at(1);
	info.outW = outW;
	info.outH = outH;

	float scaleW = static_cast<float>(src.cols) / static_cast<float>(outW);
	float scaleH = static_cast<float>(src.rows) / static_cast<float>(outH);
	float scale = std::max(scaleW, scaleH);
	if (scale <= 0.0f) scale = 1.0f;
	int newW = static_cast<int>(std::round(src.cols / scale));
	int newH = static_cast<int>(std::round(src.rows / scale));
	cv::Mat resized;
	cv::resize(rgb, resized, cv::Size(newW, newH));
	cv::Mat padded = cv::Mat::zeros(outH, outW, CV_8UC3);
	resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));
	info.scale = scale;
	info.padX = 0;
	info.padY = 0;
	return padded;
}

void YOLO::blobFromImageCHW01(const cv::Mat& rgb, float* blobPtr) const {
	const int channels = 3;
	const int imgHeight = rgb.rows;
	const int imgWidth = rgb.cols;
	for (int c = 0; c < channels; ++c) {
		for (int h = 0; h < imgHeight; ++h) {
			const cv::Vec3b* rowPtr = rgb.ptr<cv::Vec3b>(h);
			for (int w = 0; w < imgWidth; ++w) {
				blobPtr[c * imgWidth * imgHeight + h * imgWidth + w] =
					static_cast<float>(rowPtr[w][c]) / 255.0f;
			}
		}
	}
}

void YOLO::postprocessDetections(const float* output, const std::vector<int64_t>& outputShape,
	const PreprocessInfo& pp, std::vector<DL_RESULT>& results) const {
	if (modelType != YOLO_DETECT_V8 && modelType != YOLO_DETECT_V8_HALF) return;
	// 兼容两种常见输出：[1,84,S] 或 [1,S,84]
	int64_t dimA = outputShape[outputShape.size() - 2];
	int64_t dimB = outputShape[outputShape.size() - 1];
	int64_t signalResultNum; // 应为 4 + numClasses（通常 84）
	int64_t strideNum;       // 应为网格数量（通常 8400）
	bool needTranspose = false;
	if (dimA <= 256 && dimB > dimA) {
		// 形如 [84, S]
		signalResultNum = dimA;
		strideNum = dimB;
		needTranspose = true; // 我们希望 raw 按 [S,84] 行访问
	} else if (dimB <= 256 && dimA > dimB) {
		// 形如 [S, 84]
		signalResultNum = dimB;
		strideNum = dimA;
		needTranspose = false;
	} else {
		// 回退假设：最后一维为 signal
		signalResultNum = dimB;
		strideNum = dimA;
		needTranspose = false;
	}

	cv::Mat raw(signalResultNum, strideNum, CV_32F, const_cast<float*>(output));
	if (needTranspose) {
		raw = raw.t(); // [S,84]
	}
	float* data = reinterpret_cast<float*>(raw.data); // 每行一个候选，长度=signalResultNum

	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (int i = 0; i < strideNum; ++i) {
		float* classesScores = data + 4;
		if (!classes.empty()) {
			cv::Mat scores(1, static_cast<int>(classes.size()), CV_32FC1, classesScores);
			cv::Point classId;
			double maxScore;
			cv::minMaxLoc(scores, 0, &maxScore, 0, &classId);
			if (maxScore > rectConfidenceThreshold) {
				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];
				int left = static_cast<int>(std::round((x - 0.5f * w) * pp.scale)) - pp.padX;
				int top = static_cast<int>(std::round((y - 0.5f * h) * pp.scale)) - pp.padY;
				int width = static_cast<int>(std::round(w * pp.scale));
				int height = static_cast<int>(std::round(h * pp.scale));
				classIds.push_back(classId.x);
				confidences.push_back(static_cast<float>(maxScore));
				boxes.emplace_back(left, top, width, height);
			}
		}
		else {
			float maxScore = 0.0f;
			int maxIdx = 0;
			for (int c = 4; c < signalResultNum; ++c) {
				if (data[c] > maxScore) {
					maxScore = data[c];
					maxIdx = c - 4;
				}
			}
			if (maxScore > rectConfidenceThreshold) {
				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];
				int left = static_cast<int>(std::round((x - 0.5f * w) * pp.scale)) - pp.padX;
				int top = static_cast<int>(std::round((y - 0.5f * h) * pp.scale)) - pp.padY;
				int width = static_cast<int>(std::round(w * pp.scale));
				int height = static_cast<int>(std::round(h * pp.scale));
				classIds.push_back(maxIdx);
				confidences.push_back(maxScore);
				boxes.emplace_back(left, top, width, height);
			}
		}
		data += signalResultNum;
	}

	std::vector<int> nmsIndices;
	cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsIndices);
	results.clear();
	results.reserve(nmsIndices.size());
	for (int idx : nmsIndices) {
		DL_RESULT r;
		r.classId = classIds[idx];
		r.confidence = confidences[idx];
		r.box = boxes[idx];
		results.push_back(r);
	}
}

void YOLO::postprocessDetectionsBatch(const float* output, const std::vector<int64_t>& outputShape,
	const std::vector<PreprocessInfo>& ppInfos, std::vector<std::vector<DL_RESULT>>& batchResults) const {
	int64_t batch = outputShape[0];
	// 兼容 [N,84,S] 或 [N,S,84]
	int64_t dimA = outputShape[1];
	int64_t dimB = outputShape[2];
	int64_t signalResultNum = (dimA <= 256 && dimB > dimA) ? dimA : dimB;
	int64_t strideNum = (signalResultNum == dimA) ? dimB : dimA;
	size_t perSampleCount = static_cast<size_t>(signalResultNum * strideNum);
	batchResults.clear();
	batchResults.resize(static_cast<size_t>(batch));
	for (int64_t b = 0; b < batch; ++b) {
		const float* sample = output + b * perSampleCount;
		// 构造与单样本一致的形状描述
		std::vector<int64_t> singleShape = { 1, signalResultNum, strideNum };
		postprocessDetections(sample, singleShape, ppInfos[static_cast<size_t>(b)], batchResults[static_cast<size_t>(b)]);
	}
}

bool YOLO::infer(const cv::Mat& imageBgr, std::vector<DL_RESULT>& results) {
	if (!session) return false;
	PreprocessInfo pp;
	cv::Mat input = preprocessLetterboxBgrToRgb(imageBgr, imgSize, pp);
	// 使用 OpenCV 优化的打包函数生成 CHW、归一化后的 float 张量
	cv::Mat blobMat;
	cv::dnn::blobFromImage(
		input,            // 已经是 RGB 且已按 letterbox 填充
		blobMat,
		1.0 / 255.0,      // 归一化
		cv::Size(),       // 保持原尺寸
		cv::Scalar(),     // 不减均值
		/*swapRB=*/false, // 已经是 RGB，无需再交换
		/*crop=*/false,
		CV_32F);

	std::vector<int64_t> inputDims = { 1, 3, input.rows, input.cols };

	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		reinterpret_cast<float*>(blobMat.data),
		static_cast<size_t>(blobMat.total()),
		inputDims.data(), inputDims.size());

	std::vector<const char*> inNames, outNames;
	inNames.reserve(inputNodeNames.size());
	for (auto& n : inputNodeNames) inNames.push_back(n.c_str());
	outNames.reserve(outputNodeNames.size());
	for (auto& n : outputNodeNames) outNames.push_back(n.c_str());

	auto outputTensors = session->Run(options, inNames.data(), &inputTensor, 1, outNames.data(), static_cast<size_t>(outNames.size()));
	if (outputTensors.empty()) return false;

	Ort::TypeInfo typeInfo = outputTensors.front().GetTypeInfo();
	auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> outputShape = tensorInfo.GetShape();
	const float* output = outputTensors.front().GetTensorData<float>();

	if (outputShape.size() == 3) {
		postprocessDetections(output, outputShape, pp, results);
	}
	else if (outputShape.size() == 4) {
		std::vector<int64_t> viewShape{ outputShape[1], outputShape[2], outputShape[3] };
		postprocessDetections(output, viewShape, pp, results);
	}
	else {
		return false;
	}
	return true;
}

bool YOLO::inferBatch(const std::vector<cv::Mat>& imagesBgr, std::vector<std::vector<DL_RESULT>>& batchResults) {
	if (!session) return false;
	if (imagesBgr.empty()) {
		batchResults.clear();
		return true;
	}

	std::vector<PreprocessInfo> infos;
	std::vector<cv::Mat> inputs;
	inputs.reserve(imagesBgr.size());
	infos.reserve(imagesBgr.size());
	for (const auto& img : imagesBgr) {
		PreprocessInfo pp;
		inputs.emplace_back(preprocessLetterboxBgrToRgb(img, imgSize, pp));
		infos.emplace_back(pp);
	}
	const int H = inputs[0].rows;
	const int W = inputs[0].cols;
	const size_t batch = inputs.size();
	// 使用 OpenCV 一次性将一批 RGB 图像打包为 NxCxHxW float
	cv::Mat blobMat;
	cv::dnn::blobFromImages(
		inputs,           // 预处理后 RGB
		blobMat,
		1.0 / 255.0,
		cv::Size(),
		cv::Scalar(),
		/*swapRB=*/false, // 已经是 RGB
		/*crop=*/false,
		CV_32F);
	std::vector<int64_t> inputDims = { static_cast<int64_t>(batch), 3, H, W };
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		reinterpret_cast<float*>(blobMat.data),
		static_cast<size_t>(blobMat.total()),
		inputDims.data(), inputDims.size());

	std::vector<const char*> inNames, outNames;
	inNames.reserve(inputNodeNames.size());
	for (auto& n : inputNodeNames) inNames.push_back(n.c_str());
	outNames.reserve(outputNodeNames.size());
	for (auto& n : outputNodeNames) outNames.push_back(n.c_str());

	auto outputTensors = session->Run(options, inNames.data(), &inputTensor, 1, outNames.data(), static_cast<size_t>(outNames.size()));
	if (outputTensors.empty()) return false;
	Ort::TypeInfo typeInfo = outputTensors.front().GetTypeInfo();
	auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> outputShape = tensorInfo.GetShape(); // [N,84,S]
	const float* output = outputTensors.front().GetTensorData<float>();
	if (outputShape.size() != 3 || outputShape[0] != static_cast<int64_t>(batch)) {
		return false;
	}
	postprocessDetectionsBatch(output, outputShape, infos, batchResults);
	return true;
}

cv::Mat YOLO::drawDetections(const cv::Mat& imageBgr, const std::vector<DL_RESULT>& results,
	const std::vector<std::string>& classNames, const std::vector<cv::Scalar>& colors, int thickness) {
	cv::Mat vis = imageBgr.clone();
	for (const auto& r : results) {
		cv::Scalar color = colors.empty() ? cv::Scalar(0, 0, 255) : colors[static_cast<size_t>(r.classId) % colors.size()];
		cv::rectangle(vis, r.box, color, thickness);
		std::string name = (r.classId >= 0 && static_cast<size_t>(r.classId) < classNames.size()) ? classNames[r.classId] : std::to_string(r.classId);
		std::ostringstream oss;
		oss << name << " " << std::fixed << std::setprecision(2) << r.confidence;
		int baseline = 0;
		cv::Size tsize = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
		cv::rectangle(vis, cv::Rect(r.box.x, std::max(0, r.box.y - tsize.height - 6), tsize.width + 6, tsize.height + 6), color, cv::FILLED);
		cv::putText(vis, oss.str(), cv::Point(r.box.x + 3, std::max(0, r.box.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
	}
	return vis;
}


