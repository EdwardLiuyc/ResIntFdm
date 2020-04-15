#ifndef FDM_SIFT_MATCHER_HPP_
#define FDM_SIFT_MATCHER_HPP_

#include "frame.hpp"
#include "opencv2/opencv.hpp"

#include <memory>

namespace fdm {
class SiftMatcher {
 public:
  SiftMatcher();
  ~SiftMatcher() = default;

  void Match(const std::shared_ptr<Frame>& frame1,
             const std::shared_ptr<Frame>& frame2);

  inline const std::vector<cv::DMatch>& GetGoodMatches() {
    return good_matches_;
  }

 private:
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  std::vector<cv::DMatch> good_matches_;
};

}  // namespace fdm

#endif  // FDM_SIFT_MATCHER_HPP_