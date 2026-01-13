#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <meme/meme.hpp>

namespace py = pybind11;

PYBIND11_MODULE(pymeme, m, py::mod_gil_not_used())
{
    m.doc() = "Triangle Mesh Metrics"; // optional module docstring
    m.def("get_metrics", &meme::get_metrics, "Get mesh metrics");
    m.def("get_metric_names", &meme::get_metrics_names, "Get names for all mesh metrics");
}