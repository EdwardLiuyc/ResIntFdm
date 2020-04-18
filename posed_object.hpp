#ifndef FDM_POSED_OBJECTED_HPP_
#define FDM_POSED_OBJECTED_HPP_

#include <Eigen/Eigen>

namespace fdm {
class PosedObject {
 public:
  PosedObject() {}

  inline Eigen::Vector3d TranslationInEigen() const {
    return translation_eigen_;
  }
  inline Eigen::Quaterniond RotationInEigen() const { return rotation_eigen_; }
  inline Eigen::Matrix<double, 3, 4> PoseInEigen34() const {
    return pose_eigen34_;
  }
  inline Eigen::Matrix<double, 4, 4> PoseInEigen44() const {
    return pose_eigen44_;
  }
  inline void SetPose(const Eigen::Matrix4d& pose) {
    pose_eigen44_ = pose;
    pose_eigen34_ = pose.block(0, 0, 3, 4);
    camera_in_world_eigen34_ = pose_eigen44_.inverse().block(0, 0, 3, 4);
    translation_eigen_ = pose.block(0, 3, 3, 1);
    translation_ = {translation_eigen_[0], translation_eigen_[1],
                    translation_eigen_[2]};

    rotation_eigen_ =
        Eigen::Quaterniond(Eigen::Matrix3d(pose.block(0, 0, 3, 3)));
    rotation_ = {rotation_eigen_.w(), rotation_eigen_.x(), rotation_eigen_.y(),
                 rotation_eigen_.z()};
  }
  inline Eigen::Matrix<double, 3, 4> WorldInCamera() const {
    return camera_in_world_eigen34_;
  }

 protected:
  std::array<double, 3> translation_;
  std::array<double, 4> rotation_;
  Eigen::Vector3d translation_eigen_;
  Eigen::Quaterniond rotation_eigen_;
  Eigen::Matrix<double, 3, 4> pose_eigen34_;
  Eigen::Matrix<double, 4, 4> pose_eigen44_;
  Eigen::Matrix<double, 3, 4> camera_in_world_eigen34_;
};
}  // namespace fdm

#endif  // FDM_POSED_OBJECTED_HPP_