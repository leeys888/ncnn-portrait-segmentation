#include <cassert>
#include <vector>

#include <opencv2/highgui.hpp>

#include "third_parties/ncnn/src/cpu.h"

#include "ERDNet.h"
#include "timer.h"

namespace {

bool InitializeVideoCapture(cv::VideoCapture& video_capture,
                            const std::string& video_file) {
  video_file.empty() ? video_capture.open(0) : video_capture.open(video_file);

  if (!video_capture.isOpened())
    return false;

  return true;
}

cv::Mat GetFrame(cv::VideoCapture& video_capture, bool is_flip = false) {
  cv::Mat frame;
  video_capture >> frame;

  if (is_flip)
    cv::flip(frame, frame, 1);

  return frame;
}

void DrawHeatmap(cv::Mat& canvas,
                 const cv::Mat& background,
                 const ncnn::Mat& result) {
  if (result.empty())
    return;

  assert(canvas.total() == result.total());
  const int channels = canvas.channels();
  const int total_size = static_cast<int>(canvas.total() * channels);
  const auto& pixels = canvas.ptr<uint8_t>();

  assert(canvas.size() == background.size());
  auto* background_pixels = background.ptr<uint8_t>();

  for (int pixel_index = 0, alpha_index = 0;
       pixel_index < total_size;
       pixel_index += channels, alpha_index += 1) {
    const float alpha = result[alpha_index];
    for (int channel_index = 0;
         channel_index < channels;
         channel_index++) {
      float value =
          pixels[pixel_index + channel_index] * alpha +
          background_pixels[pixel_index + channel_index] * (1 - alpha);
      value = std::max(std::min(value, 255.f), 0.f);
      pixels[pixel_index + channel_index] = value;
    }
  }
}

void DrawLatency(cv::Mat& canvas, float latency) {
  cv::putText(canvas,
      cv::format("%.2f ms", latency),
      cv::Point(10, canvas.rows - 10),
      cv::FONT_HERSHEY_SIMPLEX,
      0.6,
      cv::Scalar(0, 0, 255, 0));
}

}  // namespace

int main(int argc, char* argv[]) {
  const bool has_video_file = argc > 1;
  const bool is_flip = !has_video_file;

  cv::VideoCapture video_capture;
  if (!InitializeVideoCapture(video_capture, has_video_file ? argv[1] : ""))
    return EXIT_FAILURE;

  ncnn::set_omp_dynamic(0);
  ncnn::set_omp_num_threads(4);
  const auto& segmentation = examples::model::CreateSegmenter();

  const auto& background = cv::imread("../../resources/background.png");

  while (true) {
    auto frame = GetFrame(video_capture, is_flip);

    if (frame.empty())
      break;

    static float latency = 0.f;
    measure_in_milli(latency) {
      const auto& result = segmentation->Run(frame);
      DrawHeatmap(frame, background, result);
    }

    DrawLatency(frame, latency);
    cv::imshow("frame", frame);

    if (cv::waitKey(10) == 27)
      break;
  }

  return EXIT_SUCCESS;
}
