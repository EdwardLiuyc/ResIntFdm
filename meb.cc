#include "meb.hpp"

#include <algorithm>
#include <iostream>

namespace fdm {

constexpr double kEps = 1.e-8;
constexpr double kStartT = 1000;

std::shared_ptr<MebSolver::MebResult> MebSolver::Solve() {
  double step = kStartT;
  double radius = std::numeric_limits<double>::max();
  double mt;
  Eigen::Vector3d origin(0., 0., 0.);
  int s = 0;
  while (step > kEps) {
    for (int i = 0; i < points_.size(); ++i) {
      if ((points_[s] - origin).norm() < (points_[i] - origin).norm()) {
        s = i;
      }
    }
    mt = (points_[s] - origin).norm();
    radius = std::min(radius, mt);

    // std::cout << radius << std::endl;

    origin += (points_[s] - origin) / kStartT * step;
    step *= 0.97;
  }
  // std::cout << z.x << ", " << z.y << ", " << z.z << std::endl;

  auto result = std::make_shared<MebResult>();
  result->radius = radius;
  result->origin = origin;
  return result;
}

}  // namespace fdm
