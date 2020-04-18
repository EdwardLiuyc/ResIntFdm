#include "map.hpp"

#include <glog/logging.h>
#include <unistd.h>

namespace fdm {

constexpr size_t kGoodMatchMinCount = 200;

Map::~Map() {
  LOG(INFO) << "map points num: " << map_points_.size() << std::endl;
}

void Map::InsertNewFrame(const cv::Mat& image) {
  auto new_frame = std::make_shared<Frame>(image, (int32_t)frames_.size());

  if (!frames_.empty()) {
    const int size = frames_.size();
    for (int i = size - 1; i >= 0; --i) {
      auto frame = frames_[i];
      sift_matcher_.Match(new_frame, frame);
      const auto good_matches = sift_matcher_.GetGoodMatches();
      std::cout << "Good match size: " << good_matches.size() << std::endl;

      if (kShowImage) {
        //-- Show detected matches
        cv::namedWindow("Good Matches", 0);
        cv::Mat img_matches;
        cv::drawMatches(new_frame->Image(), new_frame->KeyPoints(),
                        frame->Image(), frame->KeyPoints(), good_matches,
                        img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("Good Matches", img_matches);
        cv::waitKey(1);
        usleep(1000000);
      }

      if (good_matches.size() < kGoodMatchMinCount) {
        break;
      }
      UpdateConnection(new_frame, frame, good_matches);
    }
  }

  frames_.push_back(new_frame);
}

void Map::UpdateConnection(const std::shared_ptr<Frame> frame1,
                           const std::shared_ptr<Frame>& frame2,
                           const std::vector<cv::DMatch>& matches) {
  auto connected_map_points1 = frame1->ConnectedMapPoints();
  auto connected_map_points2 = frame2->ConnectedMapPoints();
  for (const auto& match : matches) {
    const int index1 = match.queryIdx;
    const int index2 = match.trainIdx;
    if (connected_map_points1.count(index1)) {
      if (connected_map_points2.count(index2)) {
        CHECK_EQ(connected_map_points1[index1], connected_map_points2[index2]);
      } else {
        connected_map_points2[index2] = connected_map_points1[index1];
        map_points_[connected_map_points1[index1]]
            .connected_frames_and_point[frame2->Id()] = index2;
      }
    } else {
      if (connected_map_points2.count(index2)) {
        connected_map_points1[index1] = connected_map_points2[index2];
        map_points_[connected_map_points2[index2]]
            .connected_frames_and_point[frame1->Id()] = index1;
      } else {
        // create a new map point
        map_points_.emplace_back();
        const int point_index = map_points_.size() - 1;
        connected_map_points1[index1] = point_index;
        connected_map_points2[index2] = point_index;
        map_points_.back().connected_frames_and_point[frame1->Id()] = index1;
        map_points_.back().connected_frames_and_point[frame2->Id()] = index2;
      }
    }
  }
}

}  // namespace fdm