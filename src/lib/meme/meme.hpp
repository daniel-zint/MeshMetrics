#pragma once

#include <Eigen/Core>
#include <vector>

namespace meme {

using VectorXd = Eigen::VectorXd;
using Vector3d = Eigen::Vector3d;

using MatrixXd = Eigen::MatrixXd;
using MatrixXi = Eigen::MatrixXi;

std::array<double, 16> get_metrics(const MatrixXd& V, const MatrixXi& F);

std::array<std::string, 16> get_metrics_names();

} // namespace meme
