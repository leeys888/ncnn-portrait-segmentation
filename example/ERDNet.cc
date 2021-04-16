#include "ERDNet.h"

#include <cassert>

namespace examples {

namespace {

const float kMeanValues[3] = {104.f, 112.f, 121.f};
const float kNormalizeValues[3] = {1.f/255.f, 1.f/255.f, 1.f/255.f};

const int kInputSize = 256;

ncnn::Mat Preprocess(const cv::Mat& mat) {
  ncnn::Mat image =
      ncnn::Mat::from_pixels_resize(mat.data,
          ncnn::Mat::PIXEL_BGR,
          mat.cols,
          mat.rows,
          kInputSize,
          kInputSize);

  image.substract_mean_normalize(kMeanValues, kNormalizeValues);

  return image;
}

ncnn::Mat Resize(const ncnn::Mat& mat,
                 int width,
                 int height) {
  ncnn::Option opt;
  opt.num_threads = 2;
  ncnn::Mat result;
  ncnn::resize_bilinear(mat, result, width, height, opt);

  return result;
}

}  // namespace

ERDNet::ERDNet()
  : ERDNet("../../models/erdnet.param",
           "../../models/erdnet.bin") {
}

ERDNet::ERDNet(const char* param_path, const char* bin_path) {
  int load_param_result = net_.load_param(param_path);
  assert(load_param_result == 0);
  int load_model_result = net_.load_model(bin_path);
  assert(load_model_result == 0);

  ncnn::Option opt;
  opt.lightmode = true;
  opt.blob_allocator = &blob_allocator_;
  opt.workspace_allocator = &workspace_allocator_;
  opt.num_threads = 2;
  opt_ = opt;

  net_.opt = opt;
}

ERDNet::~ERDNet() {
  workspace_allocator_.clear();
  blob_allocator_.clear();
}

ncnn::Mat ERDNet::Run(const cv::Mat& image) const {
  int width = image.cols;
  int height = image.rows;
  const auto& result = Forward(Preprocess(image));

  return Resize(result, width, height);
}

ncnn::Mat ERDNet::Forward(const ncnn::Mat& mat) const {
  ncnn::Extractor extractor = net_.create_extractor();
  extractor.input("input_blob1", mat);

  ncnn::Mat result;
  extractor.extract("sigmoid_blob1", result);
  return result;
}

namespace model {

std::unique_ptr<ERDNet> CreateSegmenter() {
  return std::make_unique<ERDNet>();
}

}  // namespace model

}  // namespace example