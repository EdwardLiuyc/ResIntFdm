#include "image_loader.hpp"

namespace fdm {

namespace fs = ::boost::filesystem;

ImageLoader::ImageLoader(const std::string& path, const std::string& im_ext) {
  // im_ext -- image extension, includes the dot Ex: ".png"
  fs::path p(path);
  for (auto it = fs::directory_iterator(p); it != fs::directory_iterator();
       it++) {
    if (fs::is_regular_file(*it) && it->path().extension().string() == im_ext) {
      image_files_.push_back(it->path());
    }
  }
  // sort by the timestamp
  std::sort(image_files_.begin(), image_files_.end());
}

std::string ImageLoader::GetImage() {
  if (image_files_.empty()) {
    return "";
  }
  const std::string image_filename = image_files_.front().generic_string();
  image_files_.pop_front();
  return image_filename;
}

}  // namespace fdm