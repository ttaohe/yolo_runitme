#include "YOLO.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

static cv::Mat numpy_uint8_hwc_to_mat(const py::array& arr) {
	py::buffer_info info = arr.request();
	if (info.ndim != 3) {
		throw std::runtime_error("Expect HxWxC uint8 array");
	}
	if (info.itemsize != 1) {
		throw std::runtime_error("Expect dtype=uint8");
	}
	const int h = static_cast<int>(info.shape[0]);
	const int w = static_cast<int>(info.shape[1]);
	const int c = static_cast<int>(info.shape[2]);
	if (c != 3) {
		throw std::runtime_error("Expect 3 channels (BGR)");
	}
	return cv::Mat(h, w, CV_8UC3, info.ptr, static_cast<size_t>(info.strides[0]));
}

static py::array mat_to_numpy_uint8_hwc(const cv::Mat& mat) {
	if (mat.type() != CV_8UC3) {
		throw std::runtime_error("Expect CV_8UC3");
	}
	std::vector<std::size_t> shape{ (size_t)mat.rows, (size_t)mat.cols, 3 };
	std::vector<std::size_t> strides{ (size_t)mat.step, (size_t)mat.elemSize(), 1 };
	return py::array(py::buffer_info(
		mat.data,				// Pointer to data
		sizeof(uint8_t),		// Size of one scalar
		py::format_descriptor<uint8_t>::format(), // Python struct-style format descriptor
		3,						// Number of dimensions
		shape,					// Buffer dimensions
		strides					// Strides (in bytes) for each index
	)).attr("copy")(); // return a copy to avoid referencing temporary buffer
}

struct PyYolo {
	YOLO core;

	bool load_model(const std::string& model_path, int img_w, int img_h, float conf, float iou,
		int model_type, int intra_threads, int log_level) {
		DL_INIT_PARAM p;
		p.modelPath = model_path;
		p.imgSize = { img_w, img_h };
		p.rectConfidenceThreshold = conf;
		p.iouThreshold = iou;
		p.modelType = static_cast<MODEL_TYPE>(model_type);
		p.intraOpNumThreads = intra_threads;
		p.logSeverityLevel = log_level;
		std::string err;
		return core.loadModel(p, err);
	}
	void set_classes(const std::vector<std::string>& names) {
		core.setClasses(names);
	}
	std::vector<std::string> get_classes() const {
		return core.getClasses();
	}
	bool warm_up() {
		return core.warmUp();
	}

	py::list infer(const py::array& image_hwc_uint8_bgr) {
		cv::Mat img = numpy_uint8_hwc_to_mat(image_hwc_uint8_bgr);
		std::vector<DL_RESULT> results;
		{
			py::gil_scoped_release release;
			if (!core.infer(img, results)) {
				throw std::runtime_error("infer failed");
			}
		}
		py::list out;
		for (const auto& r : results) {
			py::dict d;
			d["class_id"] = r.classId;
			d["confidence"] = r.confidence;
			py::dict b;
			b["x"] = r.box.x;
			b["y"] = r.box.y;
			b["w"] = r.box.width;
			b["h"] = r.box.height;
			d["box"] = b;
			out.append(d);
		}
		return out;
	}

	py::list infer_batch(const std::vector<py::array>& images) {
		std::vector<cv::Mat> mats;
		mats.reserve(images.size());
		for (const auto& a : images) {
			mats.emplace_back(numpy_uint8_hwc_to_mat(a));
		}
		std::vector<std::vector<DL_RESULT>> batchResults;
		{
			py::gil_scoped_release release;
			if (!core.inferBatch(mats, batchResults)) {
				throw std::runtime_error("inferBatch failed");
			}
		}
		py::list outBatch;
		for (const auto& results : batchResults) {
			py::list out;
			for (const auto& r : results) {
				py::dict d;
				d["class_id"] = r.classId;
				d["confidence"] = r.confidence;
				py::dict b;
				b["x"] = r.box.x;
				b["y"] = r.box.y;
				b["w"] = r.box.width;
				b["h"] = r.box.height;
				d["box"] = b;
				out.append(d);
			}
			outBatch.append(out);
		}
		return outBatch;
	}

	py::array draw(const py::array& image_hwc_uint8_bgr, const py::list& results) {
		cv::Mat img = numpy_uint8_hwc_to_mat(image_hwc_uint8_bgr);
		std::vector<DL_RESULT> rs;
		for (auto item : results) {
			py::dict d = py::reinterpret_borrow<py::dict>(item);
			py::dict b = py::reinterpret_borrow<py::dict>(d["box"]);
			DL_RESULT r;
			r.classId = d["class_id"].cast<int>();
			r.confidence = d["confidence"].cast<float>();
			r.box = cv::Rect(b["x"].cast<int>(), b["y"].cast<int>(), b["w"].cast<int>(), b["h"].cast<int>());
			rs.push_back(r);
		}
		cv::Mat vis = YOLO::drawDetections(img, rs, core.getClasses(), {});
		return mat_to_numpy_uint8_hwc(vis);
	}
};

PYBIND11_MODULE(yolort, m) {
	m.doc() = "YOLO Runtime bindings";

	py::class_<PyYolo>(m, "PyYolo")
		.def(py::init<>())
		.def("load_model", &PyYolo::load_model, py::arg("model_path"),
			py::arg("img_w") = 640, py::arg("img_h") = 640,
			py::arg("conf") = 0.25f, py::arg("iou") = 0.5f,
			py::arg("model_type") = (int)YOLO_DETECT_V8,
			py::arg("intra_threads") = 1,
			py::arg("log_level") = 3)
		.def("set_classes", &PyYolo::set_classes)
		.def("get_classes", &PyYolo::get_classes)
		.def("warm_up", &PyYolo::warm_up)
		.def("infer", &PyYolo::infer)
		.def("infer_batch", &PyYolo::infer_batch)
		.def("draw", &PyYolo::draw);
}


