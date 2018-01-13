#include <pybind11/pybind11.h>

#include <Eigen/Core>

#include "fast_spatial_matching.h"
#include "geometric_transforms.h"
#include <pybind11/stl.h>

namespace py = pybind11;

using geometric_burstiness::geometry::FastSpatialMatching;
using geometric_burstiness::geometry::Transformation;

using geometric_burstiness::geometry::AffineFeatureMatch;

using geometric_burstiness::geometry::FeatureGeometryAffine;
using geometric_burstiness::geometry::Similarity5DOF;


using Eigen::Matrix2f;
using Eigen::Matrix3f;
using Eigen::Vector2f;


class PyAffineFeatureMatch : public AffineFeatureMatch {
 public:
    PyAffineFeatureMatch(FeatureGeometryAffine f1, std::vector<FeatureGeometryAffine> f2, std::vector<int> ids) {
        feature1_ = f1;
        features2_ = f2;
        word_ids_ = ids;
    }
    void setFeatures2(std::vector<FeatureGeometryAffine> f2) {features2_ = f2;}
    std::vector<FeatureGeometryAffine> getFeatures2() const {return features2_;}
    void setWordIds(std::vector<int> ids) {word_ids_ = ids;}
    std::vector<int> getWordIds() const {return word_ids_;}
};

class PyFastSpatialMatching : public FastSpatialMatching<PyAffineFeatureMatch, FeatureGeometryAffine, Similarity5DOF> {
};

PYBIND11_MODULE(_fast_spatial_matching, m) {
    py::class_<Vector2f>(m, "Vector2f")
        .def(py::init<>());

    py::class_<FeatureGeometryAffine>(m, "FeatureGeometryAffine")
        .def(py::init<>())
        .def_readwrite("feature_id_", &FeatureGeometryAffine::feature_id_)
        .def("setPosition", &FeatureGeometryAffine::setPosition)
        .def_readwrite("a_", &FeatureGeometryAffine::a_)
        .def_readwrite("b_", &FeatureGeometryAffine::b_)
        .def_readwrite("c_", &FeatureGeometryAffine::c_);

    py::class_<PyAffineFeatureMatch>(m, "PyAffineFeatureMatch")
        .def(py::init<FeatureGeometryAffine, std::vector<FeatureGeometryAffine>, std::vector<int>>())
        // The feature in the first image.
        .def_readwrite("feature1_", &PyAffineFeatureMatch::feature1_)
        .def_property("features2_", &PyAffineFeatureMatch::getFeatures2, &PyAffineFeatureMatch::setFeatures2)
        // The indices of the visual words corresponding to the matching features in
        // features2_ (given in the same ordering).
        .def_property("word_ids_", &PyAffineFeatureMatch::getWordIds, &PyAffineFeatureMatch::setWordIds);


    py::class_<Transformation>(m, "Transformation")
        .def(py::init<>())
        .def_readwrite("A_12", &Transformation::A_12)
        .def_readwrite("t_12", &Transformation::t_12)
        .def_readwrite("A_21", &Transformation::A_21)
        .def_readwrite("t_21", &Transformation::t_21);

    py::class_<PyFastSpatialMatching>(m, "PyFastSpatialMatching")
            .def(py::init<> ())
            .def("PerformSpatialVerification", &PyFastSpatialMatching::PerformSpatialVerification);
}
