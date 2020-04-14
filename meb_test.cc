#include "meb.hpp"

#include <iostream>

int main(int argc, char** argv) {
  fdm::MebSolver solver;
  solver.InsertPoint(Eigen::Vector3d(1, 0, 0));
  solver.InsertPoint(Eigen::Vector3d(1, 0, 0));
  solver.InsertPoint(Eigen::Vector3d(1, 0, 1));
  solver.InsertPoint(Eigen::Vector3d(1, 0, -1));
  solver.InsertPoint(Eigen::Vector3d(1, 1, 0));
  solver.InsertPoint(Eigen::Vector3d(1, -1, 0));
  solver.InsertPoint(Eigen::Vector3d(2, 0, 0));
  solver.InsertPoint(Eigen::Vector3d(0, 0, 0));

  auto result = solver.Solve();
  std::cout << result->origin.transpose() << std::endl;
  std::cout << result->radius << std::endl;
  return 0;
}