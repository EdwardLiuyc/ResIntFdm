#ifndef FDM_MAP_HPP_
#define FDM_MAP_HPP_

#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>
#include <vector>
#include "frame.hpp"
#include "sift_matcher.hpp"

namespace fdm {

struct MapPoint {
  Eigen::Vector3d point;
  std::unordered_map<int, int> connected_frames_and_point;
};

class Map {
 public:
  Map() {}
  ~Map();

  void InsertNewFrame(const cv::Mat& image);

  void UpdateConnection(const std::shared_ptr<Frame> frame1,
                        const std::shared_ptr<Frame>& frame2,
                        const std::vector<cv::DMatch>& matches);

 private:
  std::vector<std::shared_ptr<Frame>> frames_;
  std::vector<MapPoint> map_points_;
  SiftMatcher sift_matcher_;
};

}  // namespace fdm

#endif  // FDM_MAP_HPP_