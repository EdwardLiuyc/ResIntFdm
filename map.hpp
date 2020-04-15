#ifndef FDM_MAP_HPP_
#define FDM_MAP_HPP_

#include <memory>
#include <vector>
#include "frame.hpp"
#include "sift_matcher.hpp"

namespace fdm {
class Map {
 public:
  Map() {}

  void InsertNewFrame(const cv::Mat& image);

 private:
  std::vector<std::shared_ptr<Frame>> frames_;
  SiftMatcher sift_matcher_;
};

}  // namespace fdm

#endif  // FDM_MAP_HPP_