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

#ifndef GEOMETRIC_BURSTINESS_SRC_FAST_SPATIAL_MATCHING_H_
#define GEOMETRIC_BURSTINESS_SRC_FAST_SPATIAL_MATCHING_H_

#include <algorithm>
#include <limits>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "geometric_transforms.h"

namespace geometric_burstiness {
namespace geometry {
using Eigen::Matrix2f;
using Eigen::Matrix3f;
using Eigen::Vector2f;

struct Transformation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // The transformation from image 1 into image 2.
  Matrix2f A_12;
  Vector2f t_12;
  // The transformation from image 2 into image 1.
  Matrix2f A_21;
  Vector2f t_21;
};

// Implements the fast spatial matching procedure from
// Philbin, Chum, Isard, Sivic, Zisserman. Object retrieval with large
// vocabularies and fast spatial matching. CPVR 2007.
// The templates are the types of features and the class used to compute the
// transformation between the images.
template <class FeatureType, class TransformationType>
class FastSpatialMatching {
 public:
  // Performs fast spatial matching. Returns the number of inliers, the
  // transformation yielding the best result (transforming features from the 1st
  // into the 2nd image), as well as the inlier matches. The latter are returned
  // as pairs, where the first entry is the index in matches, and the second
  // index gives the corresponding feature (to allow multi-matches).
  // The estimated affine transformation is given in the form A, t, such that
  // a point x1 in the first image transforms to a point x1' = A * x1 + t in the
  // second image.
  int PerformSpatialVerification(
      const std::vector<FeatureMatch<FeatureType> >& matches,
      Transformation* transform,
      std::vector<std::pair<int, int> >* inlier_ids) const {
    TransformationType transformation_provider;

    int num_matches = static_cast<int>(matches.size());
    int best_num_inliers = 0;
    // Exhaustively generates all possible transformations.
    Transformation tmp_transform;
    std::vector<std::pair<int, int> > tmp_inlier_ids;
    for (int i = 0; i < num_matches; ++i) {
      size_t num_matches_image_2 = matches[i].features2_.size();
      for (size_t m = 0; m < num_matches_image_2; ++m) {
        // Creates a transformation hypothesis.
        if (!transformation_provider.ComputeTransform(matches[i].feature1_,
                                                      matches[i].features2_[m],
                                                      &(tmp_transform.A_12),
                                                      &(tmp_transform.t_12))) {
          continue;
        }

        if (!transformation_provider.ComputeTransform(matches[i].features2_[m],
                                                      matches[i].feature1_,
                                                      &(tmp_transform.A_21),
                                                      &(tmp_transform.t_21))) {
          continue;;
        }

        int num_inliers = EvaluateHypothesis(matches, transfer_error_threshold_,
                                             scale_change_threshold_,
                                             tmp_transform, &tmp_inlier_ids);

        Eigen::Matrix<float, 2, 3> M;
        M.block<2, 2>(0, 0) = tmp_transform.A_12;
        M.col(2) = tmp_transform.t_12;

        if (num_inliers > best_num_inliers && num_inliers >= 3) {
//        if (num_inliers > best_num_inliers && num_inliers >= 4) {
          *transform = tmp_transform;
          best_num_inliers = num_inliers;
          *inlier_ids = tmp_inlier_ids;

          LocalOptimization(matches, inlier_ids, transform);
          best_num_inliers = static_cast<int>(inlier_ids->size());
        }
      }
    }
    return best_num_inliers;
  }

 private:
  // Runs local optimization as described in
  // Lebeda, Matas, Chum. Fixing the Locally Optimized RANSAC. BMVC 2012.
  // Uses the parameter settings for homography estimation proposed in the
  // paper.
  void LocalOptimization(const std::vector<FeatureMatch<FeatureType> >& matches,
                         std::vector<std::pair<int, int> >* inlier_ids,
                         Transformation* transform) const {
    // The parameter setting proposed in the paper.
    // The threshold multiplier.
    const float kMTheta = static_cast<float>(sqrt(2.0));
    // The number of inner sampling iterations.
    const int kNumReps = 10;

    // In order to avoid spending too much time on least squares estimate, we
    // use at most 7 times the minimal subset size (3), i.e., at most 21
    // matches, to estimate a least squares solution.
    std::vector<int> sample;
    sample.reserve(21);
    std::default_random_engine rng;

    // Computes a first least squares estimate.
    std::vector<std::pair<int, int> > tmp_inlier_ids;
    int num_inliers = EvaluateHypothesis(matches,
                                         kMTheta * transfer_error_threshold_,
                                         kMTheta * scale_change_threshold_,
                                         *transform, &tmp_inlier_ids);
    if (num_inliers < 4) return;
//    if (num_inliers < 3) return;

    Transformation T;
    {
      DrawRandomSample(std::min(21, num_inliers), num_inliers, &rng, &sample);
      if (!LeastSquaresEstimate(matches, tmp_inlier_ids, sample, &T)) return;

      num_inliers = EvaluateHypothesis(matches,
                                       kMTheta * transfer_error_threshold_,
                                       kMTheta * scale_change_threshold_, T,
                                       &tmp_inlier_ids);
      if (num_inliers < 4) return;
    }

    // The size of the inner samples. Notice that we want to use at least 4
    // matches to compute the affine transformation as we are only interested
    // in finding transformations with at least 4 inliers.
    const int s_is = std::max(std::min(12, num_inliers / 2), 4);

    for (int r = 0; r < kNumReps; ++r) {
      DrawRandomSample(s_is, num_inliers, &rng, &sample);

      if (!LeastSquaresEstimate(matches, tmp_inlier_ids, sample, &T)) continue;

      IterativeLeastSquares(matches, T, &rng, inlier_ids, transform);
    }
  }

  // Evaluates a transformation hypothesis and returns the number of inliers
  // as well as the indices of the inliers.
  int EvaluateHypothesis(const std::vector<FeatureMatch<FeatureType> >& matches,
                         float transfer_error_treshold,
                         float scale_change_threshold,
                         const Transformation& T,
                         std::vector<std::pair<int, int> >* inlier_ids) const {
    inlier_ids->clear();
    // Keeps track which features from the second image are already part of
    // an inlier match.
    std::unordered_set<int> match_found;
    int num_inliers = 0;
    int num_matches = static_cast<int>(matches.size());
    for (int i = 0; i < num_matches; ++i) {
      const FeatureType& f1 = matches[i].feature1_;

      float min_transfer_error = std::numeric_limits<float>::max();
      int matching_feature_index = -1;
      int feature_index = -1;
      for (const FeatureType& f2 : matches[i].features2_) {
        ++feature_index;
        if (match_found.find(f2.feature_id_) != match_found.end()) continue;

        if (!PassesScaleChangeTest(f1, f2, T, scale_change_threshold)) continue;

        float t_error = TwoWayError(f1, f2, T);
        if (t_error >= transfer_error_treshold) continue;

        if (t_error < min_transfer_error) {
          min_transfer_error = t_error;
          matching_feature_index = feature_index;
        }
      }

      if (matching_feature_index >= 0) {
        match_found.insert(
            matches[i].features2_[matching_feature_index].feature_id_);
        ++num_inliers;
        inlier_ids->push_back(std::make_pair(i, matching_feature_index));
      }
    }
    return num_inliers;
  }

  // Given a set of inliers and the subset that should be used, estimates
  // a least squares solution of the transformation. Returns false if the
  // estimate could not be compute.
  bool LeastSquaresEstimate(
      const std::vector<FeatureMatch<FeatureType> >& matches,
      const std::vector<std::pair<int, int> >& inlier_ids,
      const std::vector<int>& sample, Transformation* T) const {
    int sample_size = static_cast<int>(sample.size());
    Matrix2Xf points1(2, sample_size);
    Matrix2Xf points2(2, sample_size);

    for (int i = 0; i < sample_size; ++i) {
      const std::pair<int, int>& inlier = inlier_ids[sample[i]];
      points1.col(i) = matches[inlier.first].feature1_.x_;
      points2.col(i) = matches[inlier.first].features2_[inlier.second].x_;
    }

    if (!ComputeFullAffineTransformation(points1, points2, &(T->A_12),
                                         &(T->t_12))) {
      return false;
    }

    // Estimates the inverse of the transformation.
    Matrix3f M;
    M.setIdentity();
    M.block<2, 2>(0, 0) = T->A_12;
    M(0, 2) = (T->t_12)[0];
    M(1, 2) = (T->t_12)[1];
    Matrix3f M_inv = M.inverse();
    T->A_21 = M_inv.block<2, 2>(0, 0);
    T->t_21 = M_inv.col(2).head<2>();

    return true;
  }

  // Performs iterative least squares starting with an initial transformation.
  void IterativeLeastSquares(
      const std::vector<FeatureMatch<FeatureType> >& matches,
      const Transformation& T, std::default_random_engine* rng,
      std::vector<std::pair<int, int> >* inlier_ids,
      Transformation* transform) const {
    // The parameter setting proposed in the Lo-RANSAC paper.
    // The threshold multiplier.
    const float kMTheta = static_cast<float>(sqrt(2.0));
    // The number of iterations of least squares.
    const int kNumIters = 4;
    const float kDeltaTransferTheta = (kMTheta - 1.0f)
        * transfer_error_threshold_ / static_cast<float>(kNumIters - 1);
    const float kDeltaScaleTheta = (kMTheta - 1.0f) * scale_change_threshold_
        / static_cast<float>(kNumIters - 1);

    std::vector<std::pair<int, int> > tmp_inlier_ids;
    int num_inliers = EvaluateHypothesis(matches, transfer_error_threshold_,
                                         scale_change_threshold_, T,
                                         &tmp_inlier_ids);

    if (num_inliers >= 4 && tmp_inlier_ids.size() > inlier_ids->size()) {
      *transform = T;
      *inlier_ids = tmp_inlier_ids;
    }

    float transfer_thresh = kMTheta * transfer_error_threshold_;
    float scale_thresh = kMTheta * scale_change_threshold_;

    std::vector<int> sample;
    sample.reserve(21);
    Transformation T_tmp;
    for (int i = 0; i < kNumIters; ++i) {
      num_inliers = EvaluateHypothesis(matches, transfer_error_threshold_,
                                       scale_change_threshold_, T,
                                       &tmp_inlier_ids);
      if (num_inliers < 4) return;

      DrawRandomSample(std::min(21, num_inliers), num_inliers, rng, &sample);
      if (!LeastSquaresEstimate(matches, tmp_inlier_ids, sample, &T_tmp)) {
        return;
      }

      transfer_thresh -= kDeltaTransferTheta;
      scale_thresh -= kDeltaScaleTheta;
    }

   if (num_inliers >= 4 && tmp_inlier_ids.size() > inlier_ids->size()) {
      *transform = T_tmp;
      *inlier_ids = tmp_inlier_ids;
    }
  }

  // Draws a random sample of size k from the numbers from the interval [0, n).
  void DrawRandomSample(int k, int n, std::default_random_engine* rng,
                        std::vector<int>* sample) const {
    sample->resize(k);
    std::vector<int>& sample_ref = *sample;
    if (k == n) {
      for (int i = 0; i < k; ++i) {sample_ref[i] = i;}
      return;
    }
    std::uniform_int_distribution<int> distrib(0, n-1);
    std::default_random_engine& rng_ref = *rng;
    for (int i = 0; i < k; ++i) {
      sample_ref[i] = distrib(rng_ref);
      for (int j = 0; j < i; ++j) {
        if (sample_ref[i] == sample_ref[j]) {
          --i;
          break;
        }
      }
    }
  }

  // Returns true if the scale change between the features inside a given
  // threshold.
  bool PassesScaleChangeTest(const FeatureType& f1, const FeatureType& f2,
                             const Transformation& T, float threshold) const {
    float area_transformed = f1.GetAreaUnderTransformation(T.A_21);
    float area_measured = f2.GetArea();
    float scale_change = area_transformed / area_measured;
    if (scale_change < 1.0f) scale_change = 1.0f / scale_change;
    return scale_change < threshold;
  }

  // Returns the two-way transfer error between two features.
  float TwoWayError(const FeatureType& f1, const FeatureType& f2,
                    const Transformation& T) const {
    float error_1 = (f2.x_ - T.A_12 * f1.x_ - T.t_12).squaredNorm();
    float error_2 = (f1.x_ - T.A_21 * f2.x_ - T.t_21).squaredNorm();
    return error_1 + error_2;
  }

  // The scale threshold used to quickly reject matches. Set according to Relja
  // Arandjelovic & James Philbin.
  static constexpr float scale_change_threshold_ = 31.6227766f;  //sqrtf(1000);

  // The threshold on the two way transfer error. The parameter is set according
  // to Relja Arandjelovic & James Philbin.
  static constexpr float transfer_error_threshold_ = 40.0f * 40.0f;
};

}  // namespace geometry
}  // namespace geometric_burstiness


#endif  // GEOMETRIC_BURSTINESS_SRC_FAST_SPATIAL_MATCHING_H_
