// Copyright (c) 2016, ETH Zurich
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// T. Sattler, M. Havlena, K. Schindler, M. Pollefeys.
// Large-Scale Location Recognition And The Geometric Burstiness Problem.
// CVPR 2016.
// Author: Torsten Sattler (sattlert@inf.ethz.ch)

#ifndef GEOMETRIC_BURSTINESS_SRC_GEOMETRIC_TRANSFORMS_H_
#define GEOMETRIC_BURSTINESS_SRC_GEOMETRIC_TRANSFORMS_H_

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace geometric_burstiness {
namespace geometry {
using Eigen::Matrix2f;
using Eigen::Matrix2Xf;
using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::VectorXf;

// Represents a keypoint as its position, scale, and orientation, i.e., two
// features define a similarity transformation. Also stores the id of the
// keypoint.
class FeatureGeometrySimilarity {
 public:
  int feature_id_;
  Eigen::Vector2f x_;  // The position of the feature.
  float scale_;
  float orientation_;  // Follows VLFeats' convention of clockwise rotation.

  // Returns an approximation of the are occupied by the feature.
  float GetArea() const {
    return 1.0f / sqrtf(4.0f / (scale_ * scale_ * scale_ * scale_));
  }

  // Returns the area size of the feature under an affine transformation.
  float GetAreaUnderTransformation(const Matrix2f& A) const {
    Matrix2f M = Matrix2f::Identity();
    M /= scale_ * scale_;

    Matrix2f N = A.transpose() * M * A;

    float B = N(1, 0) + N(0, 1);
    return 1.0f / sqrtf(4.0f * N(0, 0) * N(1, 1) - B * B);
  }
};

// Represents a keypoint as its position and its affine parameters, where
// a * x^2 + b * xy + c * y^2 = 1 describes all points on the sphere. Also
// stores the id of the keypoint.
class FeatureGeometryAffine {
 public:
  int feature_id_;
  Eigen::Vector2f x_;  // The position of the feature.
  float a_;
  float b_;
  float c_;

  // Returns the Matrix mapping the ellipse into a unit circle without altering
  // the up direction and its inverse. Returns false if the transformation could
  // not be computed.
  bool GetTransformations(Matrix2f* A_to_unit_circle,
                          Matrix2f* A_from_unit_circle) const {
    if (c_ <= 0.0f) return false;

    float f = 1.0f / sqrtf(c_);
    float d = a_ - 0.5f * b_ * b_ / c_ + 0.25f * b_ * b_ / c_;
    if (d <= 0.0f) return false;
    d = 1.0f / sqrtf(d);
    float e = 0.5f * b_ * d / c_;
    *A_to_unit_circle << d, 0.0f, -e, f;

    f = sqrtf(c_);
    e = 0.5f * b_ / f;
    d = a_ - e * e;
    if (d < 0.0f) return false;
    d = sqrtf(d);

    *A_from_unit_circle << d, 0.0f, e, f;
    return true;
  }

  // Returns an approximation of the are occupied by the feature.
  float GetArea() const {
    return 1.0f / sqrtf(4.0f * a_ * c_ -  b_ * b_);
  }

  // Returns the area size of the feature under an affine transformation.
  float GetAreaUnderTransformation(const Matrix2f& A) const {
    Matrix2f M;
    M << a_, b_ * 0.5f, b_ * 0.5f, c_;

    Matrix2f N = A.transpose() * M * A;

    float a = N(0, 0);
    float b = N(1, 0) + N(0, 1);
    float c = N(1, 1);

    return 1.0f / sqrtf(4.0f * a * c -  b * b);
  }
  
  void GetTransformedEllipse(const Matrix2f& A,
                            float* a, float* b, float* c) const {
    Matrix2f M;
    M << a_, b_ * 0.5f, b_ * 0.5f, c_;
    
    Matrix2f N = A.transpose() * M * A;
    
    *a = N(0, 0);
    *b = N(1, 0) + N(0, 1);
    *c = N(1, 1);
  }
};

// Represents a 2D-2D match from the first to the second image. The feature in
// the first image can potentially match to multiple features in the second
// image. The template parameter enables the use of different kinds of features.
template <class FeatureType>
class FeatureMatch {
 public:
  FeatureType feature1_;  // The feature in the first image.
  std::vector<FeatureType> features2_;
  // The indices of the visual words corresponding to the matching features in
  // features2_ (given in the same ordering).
  std::vector<int> word_ids_;
};

typedef FeatureMatch<FeatureGeometrySimilarity> SimilarityFeatureMatch;
typedef FeatureMatch<FeatureGeometryAffine> AffineFeatureMatch;
typedef std::vector<SimilarityFeatureMatch> SimilarityFeatureMatches;
typedef std::vector<AffineFeatureMatch> AffineFeatureMatches;

// Returns true if the first feature is contained in fewer matches than the
// second one.
template <class FeatureType>
inline bool CmpFeature(const FeatureMatch<FeatureType>& match1,
		const FeatureMatch<FeatureType>& match2) {
	return match1.features2_.size() < match2.features2_.size();
}


// Given two features, both represented by a position (x, y), a rotation, and
// a scale, computes the similarity transformation transforming from the
// coordinate system of the first image into the coordinate system of the
// second image.
class Similarity3DOFPlusWithRotation {
 public:
  bool ComputeTransform(const FeatureGeometrySimilarity& feature1,
                        const FeatureGeometrySimilarity& feature2,
                        Matrix2f* sR, Vector2f* t) const {
    float scale = feature2.scale_ / feature1.scale_;
    float cos_1 = std::cos(feature1.orientation_);
    float sin_1 = std::sin(feature1.orientation_);
    float cos_2 = std::cos(feature2.orientation_);
    float sin_2 = std::sin(feature2.orientation_);

    float a = cos_1 * cos_2 + sin_1 * sin_2;
    float b = cos_1 * sin_2 - cos_2 * sin_1;
    *sR << a, b, -b, a;
    *sR *= scale;
    *t = *sR * (-feature1.x_);
    *t += feature2.x_;
    return true;
  }
};

// Same as above, only that the similarity transformation does not include a
// rotation, thus assuming that both images are upright.
class Similarity3DOF {
 public:
  bool ComputeTransform(const FeatureGeometrySimilarity& feature1,
                        const FeatureGeometrySimilarity& feature2,
                        Matrix2f* S, Vector2f* t) const {
    float scale = feature2.scale_ / feature1.scale_;
    *S << scale, 0.0f, 0.0f, scale;
    *t = feature2.x_ - scale * feature1.x_;
    return true;
  }
};

// Computes a 5DOF affine transformation between two feature regions such that
// the y-direction remains unchanges. Thus assumes that both images are (more or
// less) upright.
class Similarity5DOF {
 public:
  bool ComputeTransform(const FeatureGeometryAffine& feature1,
                        const FeatureGeometryAffine& feature2,
                        Matrix2f* A, Vector2f* t) const {
    Matrix2f A1_c, A1_e, A2_c, A2_e;
    if (!feature1.GetTransformations(&A1_c, &A1_e)) {
      return false;
    }
    if (!feature2.GetTransformations(&A2_c, &A2_e)) {
      return false;
    }
    *A = A2_c * A1_e;
    *t = feature2.x_ - (*A) * feature1.x_;
    return true;
  }
};

// Given a set of features represented as the columns of a matrix, computes
// a scaling and translation such that the features are centered around the
// origin and have an average distance of sqrt(2) from the origin.
void ComputeNormalization(const Matrix2Xf& features, float* scale,
                          Vector2f* mean) {
  *mean = features.rowwise().mean();
  float s = 0.0f;
  int num_features = features.cols();
  float n = 1.0f;
  Vector2f x;
  for (int i = 0; i < num_features; ++i, n += 1.0f) {
    x = features.col(i) - *mean;
    s = s * (n - 1.0f) / n + x.norm() / n;
  }
  *scale = static_cast<float>(sqrt(2.0)) / s;
}

// Given a set of at least three 2D-2D correspondences, computes the affine
// transformation (A, t) that relates features from the first into the second
// image. Returns false if the transformation could not be computed.
bool ComputeFullAffineTransformation(const Matrix2Xf& features1,
                                     const Matrix2Xf& features2,
                                     Matrix2f* A, Vector2f* t) {
  if (features1.cols() < 3) return false;
  if (features1.cols() != features2.cols()) return false;

  // We compute the transformation from normalized data to avoid numerical
  // issues.
  Vector2f mean1, mean2;
  float scale1, scale2;
  ComputeNormalization(features1, &scale1, &mean1);
  ComputeNormalization(features2, &scale2, &mean2);

  // Sets up the linear system that we solve to obtain a least squared solution
  // for the affine transformation.
  int num_correspondences = features1.cols();
  MatrixXf C(2 * num_correspondences, 6);
  C.setZero();
  VectorXf b(2 * num_correspondences, 1);


  int index = 0;
  Vector2f x1, x2;
  for (int i = 0; i < num_correspondences; ++i, index += 2) {
    x1 = (features1.col(i) - mean1) * scale1;
    x2 = (features2.col(i) - mean2) * scale2;
    C(index, 0) = x1[0];
    C(index, 1) = x1[1];
    C(index, 2) = 1.0f;
    b[index] = x2[0];
    C(index + 1, 3) = x1[0];
    C(index + 1, 4) = x1[1];
    C(index + 1, 5) = 1.0f;
    b[index + 1] = x2[1];
  }

  // Computes the least squares solution.
  Eigen::VectorXf sol = C.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
      .solve(b);

  // Save the transformation and undo the normalization.
  *A << sol[0], sol[1], sol[3], sol[4];
  Vector2f t1(sol[2], sol[5]);
  Vector2f t2 = t1 - scale1 * (*A) * mean1;
  *t = t2 / scale2 + mean2;
  *A *= scale1 / scale2;

  return true;
}


}  // namespace geometry
}  // namespace geometric_burstiness

#endif  // GEOMETRIC_BURSTINESS_SRC_GEOMETRIC_TRANSFORMS_H_
