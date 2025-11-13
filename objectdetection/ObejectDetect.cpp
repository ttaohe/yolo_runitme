#include "YOLO.h"
#include <iostream>
#include <fstream>
#include <iomanip>

static std::string get_base_name_no_ext(const std::string& path) {
	size_t lastSlashPos = path.find_last_of("/\\");
	size_t lastDotPos = path.find_last_of('.');
	if (lastSlashPos == std::string::npos) lastSlashPos = 0; else lastSlashPos += 1;
	if (lastDotPos == std::string::npos || lastDotPos < lastSlashPos) {
		return path.substr(lastSlashPos);
	}
	return path.substr(lastSlashPos, lastDotPos - lastSlashPos);
}

static std::string get_ext_with_dot(const std::string& path) {
	size_t lastSlashPos = path.find_last_of("/\\");
	size_t lastDotPos = path.find_last_of('.');
	if (lastDotPos == std::string::npos || (lastSlashPos != std::string::npos && lastDotPos < lastSlashPos)) {
		return std::string("");
	}
	return path.substr(lastDotPos);
}

void run_model(
	std::string &img_path,
	std::string &out_path,
	std::string &model_path,
	float conf
) {
	YOLO yolo;
	DL_INIT_PARAM params;
	params.rectConfidenceThreshold = conf;
	params.iouThreshold = 0.5f;
	params.modelPath = model_path;
	params.imgSize = { 640, 640 };
	params.modelType = YOLO_DETECT_V8;
	params.cudaEnable = false;

	std::string err;
	if (!yolo.loadModel(params, err)) {
		std::cerr << "[ERROR] loadModel failed: " << err << std::endl;
		return;
	}
	yolo.setClasses({
		"small-vehicle", "large-vehicle", "plane", "storage-tank", "ship", "harbor", "ground-track-field",
		"soccer-ball-field", "tennis-court", "swimming-pool", "baseball-diamond", "roundabout",
		"basketball-court", "bridge", "helicopter"
	});
	(void)yolo.warmUp();

	cv::Mat img = cv::imread(img_path);
	if (img.empty()) {
		std::cerr << "[ERROR] failed to read image: " << img_path << std::endl;
		return;
	}
	if (img.channels() == 1) {
		cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	}

	std::vector<DL_RESULT> results;
	if (!yolo.infer(img, results)) {
		std::cerr << "[ERROR] inference failed." << std::endl;
		return;
	}

	std::string base = get_base_name_no_ext(img_path);
	std::string ext = get_ext_with_dot(img_path);
	std::string out_img = out_path + "/" + base + "_res" + ext;
	std::string out_txt = out_path + "/" + base + "_res.txt";

	// 保存txt（YOLO格式：class x_center y_center w h confidence）
	std::ofstream ofs(out_txt);
	if (!ofs.is_open()) {
		std::cerr << "[ERROR] cannot open output file: " << out_txt << std::endl;
		return;
	}
	for (const auto& r : results) {
		float x_center = (r.box.x + r.box.width / 2.0f) / static_cast<float>(img.cols);
		float y_center = (r.box.y + r.box.height / 2.0f) / static_cast<float>(img.rows);
		float w = r.box.width / static_cast<float>(img.cols);
		float h = r.box.height / static_cast<float>(img.rows);
		ofs << r.classId << " "
			<< std::fixed << std::setprecision(6) << x_center << " "
			<< std::fixed << std::setprecision(6) << y_center << " "
			<< std::fixed << std::setprecision(6) << w << " "
			<< std::fixed << std::setprecision(6) << h << " "
			<< std::fixed << std::setprecision(6) << r.confidence << std::endl;
	}
	ofs.close();

	cv::Mat vis = YOLO::drawDetections(img, results, yolo.getClasses(), {});
	if (!cv::imwrite(out_img, vis)) {
		std::cerr << "[WARN] failed to save image: " << out_img << std::endl;
	} else {
		std::cout << "Saved: " << out_img << std::endl;
	}
	std::cout << "Saved: " << out_txt << std::endl;
}

int main(int argc, char* argv[]) {
	if (argc != 5) {
		std::cerr << "Usage: " << argv[0] << " <img_path> <out_path> <model_path> <conf_thresh>" << std::endl;
		return 1;
	}

	std::string img_path = argv[1];
	std::string out_path = argv[2];
	std::string model_path = argv[3];
	float conf = std::stof(argv[4]);

	run_model(
		img_path,
		out_path,
		model_path,
		conf
	);

	return 0;
}
