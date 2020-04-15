#ifndef FDM_IMAGE_LOADER_HPP
#define FDM_IMAGE_LOADER_HPP

#include <boost/filesystem.hpp>
#include <deque>
#include <string>

namespace fdm {

class ImageLoader {
 public:
  ImageLoader(const std::string& path, const std::string& im_ext);

  std::string GetImage();

 private:
  std::deque<boost::filesystem::path> image_files_;
};

}  // namespace fdm

#endif /* FDM_IMAGE_LOADER_HPP */