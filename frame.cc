#include "frame.hpp"

#include <opencv2/xfeatures2d/nonfree.hpp>
#ifdef _USE_CUDA_
#include "third_parties/CudaSift/cudaImage.h"
#include "third_parties/CudaSift/cudaSift.h"
#endif

namespace fdm {

constexpr int kKeyPointsNum = 3000;

Frame::Frame(const cv::Mat& image, const int32_t id) : id_(id) {
  image.copyTo(image_);

#ifdef _USE_CUDA_
  // Cuda Sift
  /* Reserve memory space for a whole bunch of SIFT features. */
  SiftData siftData;
  InitSiftData(siftData, kKeyPointsNum, true, true);
  /* Read image using OpenCV and convert to floating point. */
  cv::Mat limg;
  image.convertTo(limg, CV_32FC1);
  /* Allocate 1280x960 pixel image with device side pitch of 1280 floats. */
  /* Memory on host side already allocated by OpenCV is reused.           */
  CudaImage img;
  img.Allocate(image.cols, image.rows, image.cols, false, NULL,
               (float*)limg.data);
  /* Download image from host to device */
  img.Download();

  const int numOctaves = 5; /* Number of octaves in Gaussian pyramid */
  const float initBlur =
      1.0f; /* Amount of initial Gaussian blurring in standard deviations */
  const float thresh =
      3.5f; /* Threshold on difference of Gaussians for feature pruning */
  const float minScale =
      0.0f; /* Minimum acceptable scale to remove fine-scale features */
  const bool upScale = false; /* Whether to upscale image before extraction */
  /* Extract SIFT features */
  ExtractSift(siftData, img, numOctaves, initBlur, thresh, minScale, upScale);

  const int key_points_num = siftData.numPts;
  descriptors_ = cv::Mat::zeros(key_points_num, 128, CV_32FC1);
  key_points_.resize(key_points_num);
  cv::parallel_for_(cv::Range(0, siftData.numPts), [&](const cv::Range& range) {
    for (int r = range.start; r < range.end; r++) {
      key_points_[r] = cv::KeyPoint(
          cv::Point2f(siftData.h_data[r].xpos, siftData.h_data[r].ypos),
          siftData.h_data[r].scale, siftData.h_data[r].orientation,
          siftData.h_data[r].score, siftData.h_data[r].subsampling,
          siftData.h_data[r].match);
    }
  });
  // Convert SiftData to Mat Descriptor
  std::vector<float> data;
  for (int i = 0; i < siftData.numPts; i++) {
    data.insert(data.end(), siftData.h_data[i].data,
                siftData.h_data[i].data + 128);
  }

  cv::Mat tempDescriptor(siftData.numPts, 128, CV_32FC1, &data[0]);
  // descriptors = tempDescriptor; // Buggy!
  tempDescriptor.copyTo(descriptors_);  // Inefficient!
  FreeSiftData(siftData);
#else

  auto sift_detector =
      cv::xfeatures2d::SiftFeatureDetector::create(kKeyPointsNum);
  sift_detector->detectAndCompute(image, cv::Mat(), key_points_, descriptors_);

#endif

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

}  // namespace fdm