#include "meme.hpp"

#include <igl/edges.h>

namespace meme {

enum Metrics {
    min_min_angle = 0,
    max_min_angle = 1,
    avg_min_angle = 2,
    min_max_angle = 3,
    max_max_angle = 4,
    avg_max_angle = 5,
    min_ratio = 6,
    max_ratio = 7,
    avg_ratio = 8,
    min_shape = 9,
    max_shape = 10,
    avg_shape = 11,
    min_edge = 12,
    max_edge = 13,
    avg_edge = 14,
    num_f = 15,
    num_v = 16,
    has_zero_area = 17,
    has_zero_edge = 18
};

enum MetricsPerTri {
    min_angle = 0,
    max_angle = 1,
    ratio = 2,
    shape = 3,
};

double law_of_cosines(const double& a, const double& b, const double& c)
{
    double x = (b * b + c * c - a * a) / (2 * b * c);
    x = std::clamp(x, -1.0, 1.0);
    return std::acos(x) * (180.0 / M_PI);
}

std::array<double, 19> get_metrics(const MatrixXd& V, const MatrixXi& F)
{
    if (F.cols() != 3) {
        throw("F has not the expected number of cols. F.cols() = " + F.cols());
    }

    std::array<double, 19> metrics;
    for (double& m : metrics) {
        m = 0;
    }
    metrics[Metrics::min_min_angle] = std::numeric_limits<double>::max();
    metrics[Metrics::min_max_angle] = std::numeric_limits<double>::max();
    metrics[Metrics::min_ratio] = std::numeric_limits<double>::max();
    metrics[Metrics::min_shape] = std::numeric_limits<double>::max();
    metrics[Metrics::min_edge] = std::numeric_limits<double>::max();

    metrics[Metrics::num_f] = F.rows();
    metrics[Metrics::num_v] = V.rows();

    if (F.rows() == 0) {
        return metrics;
    }

    for (size_t i = 0; i < F.rows(); ++i) {
        const Vector3d& v0 = V.row(F(i, 0));
        const Vector3d& v1 = V.row(F(i, 1));
        const Vector3d& v2 = V.row(F(i, 2));
        const double a = (v1 - v0).norm();
        const double b = (v2 - v1).norm();
        const double c = (v0 - v2).norm();

        if (a == 0 || b == 0 || c == 0) {
            metrics[Metrics::has_zero_edge] = 1;
            continue;
        }

        std::array<double, 3> angles = {
            law_of_cosines(a, b, c),
            law_of_cosines(b, a, c),
            law_of_cosines(c, a, b)};
        const double min_angle = std::min(angles[0], std::min(angles[1], angles[2]));
        const double max_angle = std::max(angles[0], std::max(angles[1], angles[2]));

        metrics[Metrics::min_min_angle] = std::min(metrics[Metrics::min_min_angle], min_angle);
        metrics[Metrics::max_min_angle] = std::max(metrics[Metrics::max_min_angle], min_angle);
        metrics[Metrics::avg_min_angle] += min_angle;
        metrics[Metrics::min_max_angle] = std::min(metrics[Metrics::min_max_angle], max_angle);
        metrics[Metrics::max_max_angle] = std::max(metrics[Metrics::max_max_angle], max_angle);
        metrics[Metrics::avg_max_angle] += max_angle;

        const double s = (a + b + c) * 0.5;
        const double area = std::sqrt(
            std::clamp(s * (s - a) * (s - b) * (s - c), 0.0, std::numeric_limits<double>::max()));

        if (area == 0 || s == 0) {
            metrics[Metrics::has_zero_area] = 1;
            continue;
        }

        const double inradius = area / s;
        const double circumradius = (a * b * c) / (4.0 * area);
        const double radius_ratio = 2.0 * inradius / circumradius;
        const double shape_quality = (4.0 * std::sqrt(3) * area) / (a * a + b * b + c * c);

        metrics[Metrics::min_ratio] = std::min(metrics[Metrics::min_ratio], radius_ratio);
        metrics[Metrics::max_ratio] = std::max(metrics[Metrics::max_ratio], radius_ratio);
        metrics[Metrics::avg_ratio] += radius_ratio;

        metrics[Metrics::min_shape] = std::min(metrics[Metrics::min_shape], shape_quality);
        metrics[Metrics::max_shape] = std::max(metrics[Metrics::max_shape], shape_quality);
        metrics[Metrics::avg_shape] += shape_quality;

        metrics[Metrics::min_edge] =
            std::min(metrics[Metrics::min_edge], std::min(a, std::min(b, c)));
        metrics[Metrics::max_edge] =
            std::max(metrics[Metrics::max_edge], std::max(a, std::max(b, c)));
        metrics[Metrics::avg_edge] += (a + b + c);
    }

    metrics[Metrics::avg_min_angle] /= F.rows();
    metrics[Metrics::avg_max_angle] /= F.rows();
    metrics[Metrics::avg_ratio] /= F.rows();
    metrics[Metrics::avg_shape] /= F.rows();
    metrics[Metrics::avg_edge] /= F.rows() * 3;

    // edges relative to bbox
    const auto bbox_max = V.colwise().maxCoeff();
    const auto bbox_min = V.colwise().minCoeff();
    const double diag = (bbox_max - bbox_min).norm();

    metrics[Metrics::min_edge] /= diag;
    metrics[Metrics::max_edge] /= diag;
    metrics[Metrics::avg_edge] /= diag;

    return metrics;
}

MatrixXd get_metrics_per_tri(const MatrixXd& V, const MatrixXi& F)
{
    if (F.cols() != 3) {
        throw("F has not the expected number of cols. F.cols() = " + F.cols());
    }

    MatrixXd metrics;
    metrics.resize(F.rows(), 4);
    metrics.fill(0);
    metrics.row(MetricsPerTri::max_angle).fill(180);

    if (F.rows() == 0) {
        return metrics;
    }

    for (size_t i = 0; i < F.rows(); ++i) {
        const Vector3d& v0 = V.row(F(i, 0));
        const Vector3d& v1 = V.row(F(i, 1));
        const Vector3d& v2 = V.row(F(i, 2));
        const double a = (v1 - v0).norm();
        const double b = (v2 - v1).norm();
        const double c = (v0 - v2).norm();

        if (a == 0 || b == 0 || c == 0) {
            continue;
        }

        std::array<double, 3> angles = {
            law_of_cosines(a, b, c),
            law_of_cosines(b, a, c),
            law_of_cosines(c, a, b)};
        const double min_angle = std::min(angles[0], std::min(angles[1], angles[2]));
        const double max_angle = std::max(angles[0], std::max(angles[1], angles[2]));

        metrics(i, MetricsPerTri::min_angle) = min_angle;
        metrics(i, MetricsPerTri::max_angle) = max_angle;

        const double s = (a + b + c) * 0.5;
        const double area = std::sqrt(
            std::clamp(s * (s - a) * (s - b) * (s - c), 0.0, std::numeric_limits<double>::max()));

        if (area == 0 || s == 0) {
            continue;
        }

        const double inradius = area / s;
        const double circumradius = (a * b * c) / (4.0 * area);
        const double radius_ratio = 2.0 * inradius / circumradius;
        const double shape_quality = (4.0 * std::sqrt(3) * area) / (a * a + b * b + c * c);

        metrics(i, MetricsPerTri::ratio) = radius_ratio;
        metrics(i, MetricsPerTri::shape) = shape_quality;
    }

    return metrics;
}

VectorXd get_relative_edge_lengths(const MatrixXd& V, const MatrixXi& F)
{
    if (F.cols() != 3) {
        throw("F has not the expected number of cols. F.cols() = " + F.cols());
    }

    // edges relative to bbox
    const auto bbox_max = V.colwise().maxCoeff();
    const auto bbox_min = V.colwise().minCoeff();
    const double diag = (bbox_max - bbox_min).norm();
    const double inv_diag = 1. / diag;

    MatrixXi E;
    igl::edges(F, E);

    VectorXd lengths;
    lengths.resize(E.rows());

    for (size_t i = 0; i < E.rows(); ++i) {
        const Vector3d& p0 = V.row(E(i, 0));
        const Vector3d& p1 = V.row(E(i, 1));
        const double l = (p1 - p0).norm();
        lengths[i] = l * inv_diag;
    }

    return lengths;
}

std::array<std::string, 19> get_metrics_names()
{
    return std::array<std::string, 19>{
        "min_min_angle",
        "max_min_angle",
        "avg_min_angle",
        "min_max_angle",
        "max_max_angle",
        "avg_max_angle",
        "min_ratio",
        "max_ratio",
        "avg_ratio",
        "min_shape",
        "max_shape",
        "avg_shape",
        "min_edge",
        "max_edge",
        "avg_edge",
        "#F",
        "#V",
        "has_zero_area",
        "has_zero_edge"};
}

std::array<std::string, 4> get_metrics_names_per_tri()
{
    return std::array<std::string, 4>{"min_angle", "max_angle", "ratio", "shape"};
}

} // namespace meme