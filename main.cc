#include "image_loader.hpp"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "map.hpp"

#ifdef _USE_CUDA_
#include "third_parties/CudaSift/cudaSift.h"
#endif

int main(int argc, char** argv) {
  fdm::ImageLoader loader("/mnt/Data/sfm/lund_cath_small/Cathedral", ".jpg");
  std::string image_file;

#ifdef _USE_CUDA_
  InitCuda(0);
#endif

  auto sift_detector = cv::xfeatures2d::SiftFeatureDetector::create(3000);
  fdm::Map map;
  while (!(image_file = loader.GetImage()).empty()) {
    std::cout << image_file << std::endl;
    cv::Mat image = cv::imread(image_file, 0);
    map.InsertNewFrame(image);
    cv::waitKey(0);
  }
  return 0;
}