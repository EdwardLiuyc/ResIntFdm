#include "meb.hpp"
#include "gtest/gtest.h"

#include <iostream>

TEST(meb_test, test1) {
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
  EXPECT_NEAR(result->radius, 1., 1.e-6);
  EXPECT_NEAR(result->origin[0], 1., 1.e-6);
  EXPECT_NEAR(result->origin[1], 0., 1.e-6);
  EXPECT_NEAR(result->origin[2], 0., 1.e-6);
}

TEST(meb_test, test2) {
  fdm::MebSolver solver;
  solver.InsertPoint(Eigen::Vector3d(-1, 0, 0));
  solver.InsertPoint(Eigen::Vector3d(-1, 0, 0));
  solver.InsertPoint(Eigen::Vector3d(-1, 0, 1));
  solver.InsertPoint(Eigen::Vector3d(-1, 0, -1));
  solver.InsertPoint(Eigen::Vector3d(-1, 1, 0));
  solver.InsertPoint(Eigen::Vector3d(-1, -1, 0));
  solver.InsertPoint(Eigen::Vector3d(0, 0, 0));
  solver.InsertPoint(Eigen::Vector3d(-2, 0, 0));

  auto result = solver.Solve();
  EXPECT_NEAR(result->radius, 1., 1.e-6);
  EXPECT_NEAR(result->origin[0], -1., 1.e-6);
  EXPECT_NEAR(result->origin[1], 0., 1.e-6);
  EXPECT_NEAR(result->origin[2], 0., 1.e-6);
}