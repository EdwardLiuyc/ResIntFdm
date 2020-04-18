#include "sift_matcher.hpp"

#include <chrono>

namespace fdm {

SiftMatcher::SiftMatcher()
    : matcher_(
          cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED)) {}

void SiftMatcher::Match(const std::shared_ptr<Frame>& frame1,
                        const std::shared_ptr<Frame>& frame2) {
  std::vector<std::vector<cv::DMatch> > knn_matches;
  const auto start = std::chrono::high_resolution_clock::now();
  matcher_->knnMatch(frame1->Descriptor(), frame2->Descriptor(), knn_matches,
                     2);
  const auto end = std::chrono::high_resolution_clock::now();
  int64_t duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  std::cout << duration << " us" << std::endl;
  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;
  std::vector<cv::DMatch> lowe_filtered_matches;
  std::vector<cv::Point2f> points1, points2;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance <
        ratio_thresh * knn_matches[i][1].distance) {
      lowe_filtered_matches.push_back(knn_matches[i][0]);
      points1.push_back(frame1->KeyPoints().at(knn_matches[i][0].queryIdx).pt);
      points2.push_back(frame2->KeyPoints().at(knn_matches[i][0].trainIdx).pt);
    }
  }

  std::cout << "filtered match size: " << lowe_filtered_matches.size()
            << std::endl;

  // [[4.76529213e+03 0.00000000e+00 1.87956718e+03]
  //  [0.00000000e+00 4.75245199e+03 1.01316490e+03]
  //  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
  std::vector<uchar> fundamental_status;
  good_matches_.clear();
  // cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 0.8, 0.99,
  //                        fundamental_status);
  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = 4.75245199e3;
  K.at<float>(1, 1) = 4.76529213e3;
  K.at<float>(0, 2) = 1.01316490e3;
  K.at<float>(1, 2) = 1.87956718e3;
  cv::findEssentialMat(points1, points2, K, cv::FM_RANSAC, 0.999, 0.8,
                       fundamental_status);
  for (size_t i = 0; i < points1.size(); ++i) {
    if (fundamental_status[i] != 0) {
      good_matches_.push_back(lowe_filtered_matches[i]);
    }
  }
}

}  // namespace fdm