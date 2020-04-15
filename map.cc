#include "map.hpp"

#include "unistd.h"

namespace fdm {

constexpr size_t kGoodMatchMinCount = 200;

void Map::InsertNewFrame(const cv::Mat& image) {
  auto new_frame = std::make_shared<Frame>(image);

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

      frame->UpdateConnection(good_matches, size);
    }
  }

  frames_.push_back(new_frame);
}

}  // namespace fdm