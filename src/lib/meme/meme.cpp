#include "meme.hpp"

#include <igl/edges.h>

namespace meme {

enum Metrics {
    min_angle = 0,
    max_angle = 1,
    avg_angle = 2,
    min_ratio = 3,
    max_ratio = 4,
    avg_ratio = 5,
    min_shape = 6,
    max_shape = 7,
    avg_shape = 8,
    min_edge = 9,
    max_edge = 10,
    avg_edge = 11,
    num_f = 12,
    num_v = 13,
    has_zero_area = 14,
    has_zero_edge = 15
};

double law_of_cosines(const double& a, const double& b, const double& c)
{
    double x = (b * b + c * c - a * a) / (2 * b * c);
    x = std::clamp(x, -1.0, 1.0);
    return std::acos(x) * (180.0 / M_PI);
}

std::array<double, 16> get_metrics(const MatrixXd& V, const MatrixXi& F)
{
    std::array<double, 16> metrics;
    for (double& m : metrics) {
        m = 0;
    }
    metrics[Metrics::min_angle] = std::numeric_limits<double>::max();
    metrics[Metrics::min_ratio] = std::numeric_limits<double>::max();
    metrics[Metrics::min_shape] = std::numeric_limits<double>::max();
    metrics[Metrics::min_edge] = std::numeric_limits<double>::max();

    metrics[Metrics::num_f] = F.rows();
    metrics[Metrics::num_v] = V.rows();

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

        metrics[Metrics::min_angle] = std::min(
            metrics[Metrics::min_angle],
            std::min(angles[0], std::min(angles[1], angles[2])));
        metrics[Metrics::max_angle] = std::max(
            metrics[Metrics::max_angle],
            std::max(angles[0], std::max(angles[1], angles[2])));
        metrics[Metrics::avg_angle] += (angles[0] + angles[1] + angles[2]);

        const double s = (a + b + c) * 0.5;
        const double area = std::sqrt(
            std::clamp(s * (s - a) * (s - b) * (s - c), 0.0, std::numeric_limits<double>::max()));

        if (area == 0) {
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

    metrics[Metrics::avg_angle] /= F.rows() * 3;
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

std::array<std::string, 16> get_metrics_names()
{
    return std::array<std::string, 16>{
        "min_angle",
        "max_angle",
        "avg_angle",
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

} // namespace meme