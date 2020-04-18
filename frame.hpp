#ifndef FDM_FRAME_HPP_
#define FDM_FRAME_HPP_

#include "global_configs.hpp"
#include "opencv2/opencv.hpp"
#include "posed_object.hpp"

namespace fdm {

class Frame : public PosedObject {
 public:
  Frame(const cv::Mat& image, const int32_t id);

  inline const cv::Mat& Image() { return image_; }
  inline const cv::Mat& Descriptor() const { return descriptors_; }
  inline const std::vector<cv::KeyPoint>& KeyPoints() const {
    return key_points_;
  }

  inline int32_t Id() const { return id_; }
  inline std::unordered_map<int, int>& ConnectedMapPoints() {
    return connected_map_points_;
  }

 private:
  const int32_t id_;
  cv::Mat image_;
  cv::Mat descriptors_;
  std::vector<cv::KeyPoint> key_points_;
  std::unordered_map<int, int> connected_map_points_;
};

}  // namespace fdm

#endif