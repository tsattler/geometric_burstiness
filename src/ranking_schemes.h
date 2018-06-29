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

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "geometric_transforms.h"

#ifndef GEOMETRIC_BURSTINESS_RANKING_SCHEMES_H_
#define GEOMETRIC_BURSTINESS_RANKING_SCHEMES_H_

namespace geometric_burstiness {
namespace geometry {
// Given a set of feature matches, computes and returns the effective inlier
// count for the query image (cf. Eq. 9 in the paper).
double ComputeEffectiveInlierCount(
    const AffineFeatureMatches& matches,
    const std::vector<std::pair<int, int> >& inliers) {
  double effective_inliers = 0.0;

  // Determines the size of the matrix required to compute the area covered by
  // all inliers.
  int max_x = 0;
  int max_y = 0;
  for (const AffineFeatureMatch& match : matches) {
    max_x = std::max(max_x, static_cast<int>(match.feature1_.x_[0] + 0.5));
    max_y = std::max(max_y, static_cast<int>(match.feature1_.x_[1] + 0.5));
  }
  Eigen::MatrixXi area_query_image = Eigen::MatrixXi::Zero(max_x + 24,
                                                           max_y + 24);
  for (const std::pair<int, int>& inlier : inliers) {
    int x = static_cast<int>(matches[inlier.first].feature1_.x_[0] + 0.5f);
    int y = static_cast<int>(matches[inlier.first].feature1_.x_[1] + 0.5f);
    area_query_image.block<24, 24>(x, y) = Eigen::MatrixXi::Ones(24, 24);
  }
  double area_covered = static_cast<double>(area_query_image.sum());
  double ideal_area =
      static_cast<double>(static_cast<int>(inliers.size()) * 576);

  return area_covered / ideal_area * static_cast<double>(inliers.size());
}
}  // namespace geometry

namespace reranking {
// Comparison function.
template<typename T1, typename T2>
inline bool CmpPairFirstGreater(const std::pair<T1, T2>& a,
                                const std::pair<T1, T2>& b) {
  return a.first > b.first;
}

// Performs re-ranking based on the number of inliers. Given as set of
// database images, where each database image is represented as a (number of
// inliers, image_id) pair, sorts the database images in decreasing order of
// their numbers of inliers. A stable sorting method is used, i.e., database
// images with the same number of inliers are ranked based on the order of the
// input list.
void ReRankingInlierCount(
    const std::vector<std::pair<int, int> >& initial_ranking,
    std::vector<std::pair<int, int> >* ranked_list) {
  ranked_list->clear();
  if (initial_ranking.empty())
    return;

  ranked_list->assign(initial_ranking.begin(), initial_ranking.end());

  std::stable_sort(ranked_list->begin(), ranked_list->end(),
                   CmpPairFirstGreater<int, int>);
}

// Performs re-ranking based on the effective inlier count. Given as set of
// database images, where each database image is represented as a (effective
// inlier count value, image_id) pair, sorts the database images in decreasing
// order of the effective inlier count. A stable sorting method is used, i.e.,
// database images with the same inlier count are ranked based on the order of
// the input list.
void ReRankingEffectiveInlierCount(
    const std::vector<std::pair<double, int> >& initial_ranking,
    std::vector<std::pair<double, int> >* ranked_list) {
  ranked_list->clear();
  if (initial_ranking.empty()) return;

  ranked_list->assign(initial_ranking.begin(), initial_ranking.end());

  std::stable_sort(ranked_list->begin(), ranked_list->end(),
                   CmpPairFirstGreater<double, int>);
}

// Performs re-ranking based on the inter-image geometric burstiness count.
// The input is a list of database images together with the indices of the
// inlier features in the query image and the number of features in the query
// image. The output is a sorted list of database images in decreasing order of
// scores. A stable sorting method is used, i.e., database images with the same
// burstiness score are ranked based on the order of the input list.
void ReRankingInterImageGeometricBurstiness(
    const std::vector<std::pair<int, std::vector<int> > >& inliers_per_db,
    int num_query_features, std::vector<std::pair<double, int> >* ranked_list) {
  ranked_list->clear();

  if (num_query_features == 0) return;
  // Computes how often each feature is an inlier.
  std::vector<int> num_inlier_occurrences(num_query_features, 0);
  for (const std::pair<int, std::vector<int> >& db_image : inliers_per_db) {
    for (const int f_id : db_image.second) {
      num_inlier_occurrences[f_id] += 1;
    }
  }

  // Computes the inter-image burstiness scores.
  for (const std::pair<int, std::vector<int> >& db_image : inliers_per_db) {
    double score = 0.0;
    for (const int f_id : db_image.second) {
      double weight = num_inlier_occurrences[f_id];
      if (weight <= 0.0) continue;
      weight = 1.0 / sqrt(weight);
      score += weight;
    }
    ranked_list->push_back(std::make_pair(score, db_image.first));
  }

  std::stable_sort(ranked_list->begin(), ranked_list->end(),
                   CmpPairFirstGreater<double, int>);
}

// Performs re-ranking based on the inter-place geometric burstiness count.
// The input is a list of database images together with the indices of the
// inlier features in the query image, the number of features in the query image
// as well as the geo-coordinates of the database images. The coordinates are
// assumed to be in a coordinate system where the Euclidean distance corresponds
// to a distance in meters. Each column in db_pos corresponds to the coordinate
// of one database image. In addition, a threshold on the squared maximal
// distance between two images belonging to the same place is given.
// The function returns both the list of database images sorted based on the
// standard inter-place scheme as well as based on the popularity-weighted
// scheme. A stable sorting method is used, i.e., database images with the same
// score are ranked based on the order of the input list.
// IMPORTANT: To reproduce the results from the paper, inliers_per_db should be
//            sorted in descending number of inliers.
void ReRankingInterPlaceGeometricBurstiness(
    const std::vector<std::pair<int, std::vector<int> > >& inliers_per_db,
    int num_query_features, const Eigen::Matrix2Xd& db_pos,
    double squared_place_dist_threshold,
    std::vector<std::pair<double, int> >* inter_place_ranked_list,
    std::vector<std::pair<double, int> >* inter_place_pop_ranked_list) {
  inter_place_ranked_list->clear();
  inter_place_pop_ranked_list->clear();

  if (num_query_features == 0) return;

  ////
  // Divides the database images into a set of places as described in the paper.
  int num_db_images = static_cast<int>(inliers_per_db.size());
  if (num_db_images == 0) return;

  // The place id per database image.
  std::vector<int> place_id_per_db_image(num_db_images, 0);
  // Stores the minimal distance to its nearest place for each database image.
  std::vector<double> min_distance_to_nearest_place(
      num_db_images, std::numeric_limits<double>::max());
  // The first place is defined by the top-ranked database image, given as the
  // first image in inlier_per_db.
  min_distance_to_nearest_place[0] = 0.0;
  for (int i = 1; i < num_db_images; ++i) {
    min_distance_to_nearest_place[i] = (db_pos.col(inliers_per_db[0].first)
        - db_pos.col(inliers_per_db[i].first)).squaredNorm();
  }
  // A new place is added for each database image further than
  // squared_place_dist_threshold m^2 away from all other places.
  int num_places = 0;
  while (true) {
    ++num_places;
    // Finds the database image furthest away from all places.
    double squared_max_dist = squared_place_dist_threshold;
    int max_index = -1;
    for (int i = 1; i < num_db_images; ++i) {
      if (min_distance_to_nearest_place[i] > squared_max_dist) {
        squared_max_dist = min_distance_to_nearest_place[i];
        max_index = i;
      }
    }

    if (max_index == -1) break;

    min_distance_to_nearest_place[max_index] = 0.0;
    place_id_per_db_image[max_index] = num_places;

    // Creates a new place and updates the minimum distance for each database
    // image to its nearest place.
    for (int i = 1; i < num_db_images; ++i) {
      double dist = (db_pos.col(inliers_per_db[max_index].first)
          - db_pos.col(inliers_per_db[i].first)).squaredNorm();
      if (dist < min_distance_to_nearest_place[i]) {
        min_distance_to_nearest_place[i] = dist;
        place_id_per_db_image[i] = num_places;
      }
    }
  }

  ////
  // Computes for each query feature to how many places it is an inlier as well
  // as the set of inlier features for each place.
  std::vector<std::unordered_set<int> > inlier_ids_per_place(num_places);
  std::vector<std::unordered_set<int>
      > num_places_per_query_feature(num_query_features);

  for (int i = 0; i < num_db_images; ++i) {
    const int db_id = inliers_per_db[i].first;
    const int place_id = place_id_per_db_image[i];
    for (const int f_id : inliers_per_db[i].second) {
      num_places_per_query_feature[f_id].insert(place_id);
      inlier_ids_per_place[place_id].insert(f_id);
    }
  }

  // Computes the number of inliers for the most popular place.
  int max_num_inliers = 0;
  for (int i = 0; i < num_places; ++i) {
    max_num_inliers = std::max(
        max_num_inliers, static_cast<int>(inlier_ids_per_place[i].size()));
  }
  if (max_num_inliers == 0) return;
  // Computes the normalization constant for the popularity weighting scheme.
  double pop_constant = 1.0 / static_cast<int>(max_num_inliers);

  ////
  // Computes the two burstiness scores.
  for (int i = 0; i < num_db_images; ++i) {
    const int db_id = inliers_per_db[i].first;
    const int place_id = place_id_per_db_image[i];
    double inter_place_score = 0.0;
    for (const int f_id : inliers_per_db[i].second) {
      size_t num_relevant_places = num_places_per_query_feature[f_id].size();
      if (num_relevant_places == 0)
        continue;
      inter_place_score += 1.0 / static_cast<double>(num_relevant_places);
    }
    double pop_factor =
        static_cast<double>(inlier_ids_per_place[place_id].size())
            * pop_constant;
    inter_place_ranked_list->push_back(
        std::make_pair(inter_place_score, db_id));
    inter_place_pop_ranked_list->push_back(
        std::make_pair(inter_place_score * pop_factor, db_id));
  }

  std::stable_sort(inter_place_ranked_list->begin(),
                   inter_place_ranked_list->end(),
                   CmpPairFirstGreater<double, int>);

  std::stable_sort(inter_place_pop_ranked_list->begin(),
                   inter_place_pop_ranked_list->end(),
                   CmpPairFirstGreater<double, int>);
}

}  // namespace reranking

}  // namespace geometric burstiness

#endif  // GEOMETRIC_BURSTINESS_RANKING_SCHEMES_H_
