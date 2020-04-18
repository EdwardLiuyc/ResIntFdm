#ifndef FDM_FRAME_HPP_
#define FDM_FRAME_HPP_

#include "opencv2/opencv.hpp"

#include "global_configs.hpp"

namespace fdm {
class Map;
class Frame {
 public:
  Frame(const cv::Mat& image, Map* const map);

  inline const cv::Mat& Image() { return image_; }
  inline const cv::Mat& Descriptor() const { return descriptors_; }
  inline const std::vector<cv::KeyPoint>& KeyPoints() const {
    return key_points_;
  }

  void UpdateConnection(const std::vector<cv::DMatch>& matches,
                        int target_frame_index);

 private:
  Map* map_;
  cv::Mat image_;
  cv::Mat descriptors_;
  std::vector<cv::KeyPoint> key_points_;

  std::map<
      int /* key point index in current frame*/,
      std::vector<std::pair<int /* frame index */, int /* key point index*/>>>
      connections_;
};

}  // namespace fdm

#endif