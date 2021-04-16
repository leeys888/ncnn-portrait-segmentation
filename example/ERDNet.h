#ifndef EXAMPLES_ERDNET_H_
#define EXAMPLES_ERDNET_H_

#include <memory>

#include <opencv2/imgproc.hpp>

#include "third_parties/ncnn/src/net.h"

namespace examples {

class ERDNet {
 public:
  ERDNet();
  ERDNet(const char* param_path, const char* bin_path);

  virtual ~ERDNet();

  ncnn::Mat Run(const cv::Mat& image) const;

 private:
  ncnn::Mat Forward(const ncnn::Mat& mat) const;

 private:
  ncnn::Net net_;
  ncnn::Option opt_;
  ncnn::PoolAllocator workspace_allocator_;
  ncnn::UnlockedPoolAllocator blob_allocator_;
};

namespace model {
  std::unique_ptr<ERDNet> CreateSegmenter();

}  // namespace model

}  // namespace example

#endif  // EXAMPLES_ERDNET_H_
