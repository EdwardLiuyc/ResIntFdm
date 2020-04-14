#ifndef FDM_MEB_HPP_
#define FDM_MEB_HPP_

#include <Eigen/Eigen>
#include <memory>
#include <vector>

namespace fdm {

class MebSolver {
 public:
  MebSolver() {}
  ~MebSolver() = default;

  struct MebResult {
    Eigen::Vector3d origin;
    double radius;
  };

  inline void InsertPoint(const Eigen::Vector3d& point) {
    points_.push_back(point);
  }

  std::shared_ptr<MebResult> Solve();

 private:
  std::vector<Eigen::Vector3d> points_;
};
}  // namespace fdm

#endif  // FDM_MEB_HPP_