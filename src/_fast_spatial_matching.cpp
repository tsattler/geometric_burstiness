#include <pybind11/pybind11.h>

#include <Eigen/Core>

#include "fast_spatial_matching.h"
#include "geometric_transforms.h"

namespace py = pybind11;

using geometric_burstiness::geometry::FastSpatialMatching;
using geometric_burstiness::geometry::Transformation;

using geometric_burstiness::geometry::FeatureGeometryAffine;
using geometric_burstiness::geometry::Similarity5DOF;


using Eigen::Matrix2f;
using Eigen::Matrix3f;
using Eigen::Vector2f;

PYBIND11_MODULE(_fast_spatial_matching, m) {
    py::class_<Vector2f>(m, "Vector2f")
        .def(py::init<>());

    py::class_<FeatureGeometryAffine>(m, "FeatureGeometryAffine")
        .def(py::init<>())
        .def_readwrite("feature_id_", &Transformation::feature_id_)
        .def_readwrite("x_", &Transformation::x_)
        .def_readwrite("a_", &Transformation::a_)
        .def_readwrite("b_", &Transformation::b_)
        .def_readwrite("c_", &Transformation::c_);

    py::class_<AffineFeatureMatch>(m, "AffineFeatureMatch")
        .def(py::init<>());

    py::class_<Transformation>(m, "Transformation")
        .def(py::init<>())
        .def_readwrite("A_12", &Transformation::A_12)
        .def_readwrite("t_12", &Transformation::t_12)
        .def_readwrite("A_21", &Transformation::A_21)
        .def_readwrite("t_21", &Transformation::t_21);

    py::class_<FastSpatialMatching<FeatureGeometryAffine, Similarity5DOF>>(m, "FastSpatialMatching")
            .def(py::init<> ())
            .def("PerformSpatialVerification", &FastSpatialMatching::PerformSpatialVerification);
}
