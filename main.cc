#include "image_loader.hpp"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

int main(int argc, char** argv) {
  fdm::ImageLoader loader("/mnt/Data/sfm/lund_cath_small/Cathedral", ".jpg");
  std::string image_file;

  auto sift_detector = cv::xfeatures2d::SiftFeatureDetector::create(3000);
  cv::namedWindow("debug", 0);
  cv::namedWindow("sift", 0);
  while (!(image_file = loader.GetImage()).empty()) {
    std::cout << image_file << std::endl;

    cv::Mat image = cv::imread(image_file, 0);
    cv::imshow("debug", image);
    std::vector<cv::KeyPoint> key_points;
    cv::Mat descs;
    sift_detector->detectAndCompute(image, cv::Mat(), key_points, descs);
    cv::Mat image_with_sift;
    cv::drawKeypoints(image, key_points, image_with_sift,
                      cv::Scalar(-1, -1, -1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("sift", image_with_sift);
    cv::waitKey(0);
  }
  return 0;
}