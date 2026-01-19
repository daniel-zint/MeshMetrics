#pragma once

#include <Eigen/Core>
#include <vector>

namespace meme {

using VectorXd = Eigen::VectorXd;
using Vector3d = Eigen::Vector3d;

using MatrixXd = Eigen::MatrixXd;
using MatrixXi = Eigen::MatrixXi;

std::array<double, 19> get_metrics(const MatrixXd& V, const MatrixXi& F);

MatrixXd get_metrics_per_tri(const MatrixXd& V, const MatrixXi& F);

VectorXd get_relative_edge_lengths(const MatrixXd& V, const MatrixXi& F);

std::array<std::string, 19> get_metrics_names();

std::array<std::string, 4> get_metrics_names_per_tri();

} // namespace meme
