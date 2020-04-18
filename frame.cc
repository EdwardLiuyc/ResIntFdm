#include "frame.hpp"

#include <opencv2/xfeatures2d/nonfree.hpp>

namespace fdm {

constexpr int kKeyPointsNum = 3000;

Frame::Frame(const cv::Mat& image, Map* const map) : map_(map) {
  image.copyTo(image_);
  auto sift_detector =
      cv::xfeatures2d::SiftFeatureDetector::create(kKeyPointsNum);
  sift_detector->detectAndCompute(image, cv::Mat(), key_points_, descriptors_);

  if (kShowImage) {
    cv::namedWindow("sift", 0);
    cv::Mat image_with_sift;
    cv::drawKeypoints(image, key_points_, image_with_sift,
                      cv::Scalar(-1, -1, -1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("sift", image_with_sift);
    cv::waitKey(1);
  }
}

void Frame::UpdateConnection(const std::vector<cv::DMatch>& matches,
                             int target_frame_index) {
  for (const auto& match : matches) {
    connections_[match.queryIdx].emplace_back(target_frame_index,
                                              match.trainIdx);
  }
}

}  // namespace fdm