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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <stdint.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "inverted_file.h"
#include "inverted_index.h"
#include "ranking_schemes.h"
#include "structures.h"

// For Pittsburgh, a location is determined to be correct if the retrieved
// db image is within 25 meters of the query image.
const double gt_inlier_dist = 25.0f;
const double squared_gt_inlier_dist = gt_inlier_dist * gt_inlier_dist;

// How many of the top-ranked images should be evaluated.
const int kNumSpatial = 200;

// How many nearest neighboring words to use for soft assignments.
// It is required that exactly kNumNN words are stored in the feature files
// loaded for the query images.
const int kNumNNWords = 5;

// Compares twor (image id, vectorof inlier ids) pairs and returns true if the
// first pair has more images.
template<typename T1, typename T2>
inline bool CompareBasedOnNumberOfInliers(
    const std::pair<T1, std::vector<T2> >& a,
    const std::pair<T1, std::vector<T2> >& b) {
  return a.second.size() > b.second.size();
}

int main (int argc, char **argv)
{
  if (argc < 7) {
    std::cout << "________________________________________________________________________________________________" << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -      Queries a pre-build inverted index and writes out the results.                        - " << std::endl;
    std::cout << " -                       written 2015 by Torsten Sattler (sattlert@inf.ethz.ch)               - " << std::endl;
    std::cout << " -    copyright ETH Zurich, 2015-2016                                                         - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " - usage: query list_sifts list_db index outfile_prefix sift_type pos_db                      - " << std::endl;
    std::cout << " - Parameters:                                                                                - " << std::endl;
    std::cout << " -  list_sifts                                                                                - " << std::endl;
    std::cout << " -     A text file containing the filenames of all query files without file ending (but with  - " << std::endl;
    std::cout << " -     a trailing .).                                                                         - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  list_db                                                                                   - " << std::endl;
    std::cout << " -     A text file containing the filename of the .jpg images in the database.                - " << std::endl;
    std::cout << " -     Needs to be in the same order as list_sifts for build_partial_index.                   - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  index                                                                                     - " << std::endl;
    std::cout << " -     The inverted index, e.g., created with build_index.                                    - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  outfile_prefix                                                                            - " << std::endl;
    std::cout << " -     The prefix of the output files. This program generates 6 files:                        - " << std::endl;
    std::cout << " -     outfile_prefix.retrieval_ranking.txt, outfile_prefix.inlier_count_ranking.txt,         - " << std::endl;
    std::cout << " -     outfile_prefix.effective_inlier_count_ranking.txt,                                     - " << std::endl;
    std::cout << " -     outfile_prefix.inter_image_geometric_burstiness_ranking.txt,                           - " << std::endl;
    std::cout << " -     outfile_prefix.inter_place_geometric_burstiness_ranking.txt,                           - " << std::endl;
    std::cout << " -     outfile_prefix.inter_place_popularity_geometric_burstiness_ranking.txt,                - " << std::endl;
    std::cout << " -     each one corresponding to one re-ranking scheme. For each query, the file contains     - " << std::endl;
    std::cout << " -     a list of retrieved images ranked according to the metric. Each line of the text files - " << std::endl;
    std::cout << " -     has the format query_name.jpg database_name.jpg score, where score is the ranking      - " << std::endl;
    std::cout << " -     score computed between the query and the database image.                               - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  sift_type                                                                                 - " << std::endl;
    std::cout << " -     How the features were generated: 0 - VL_FEAT, 1 - VGG binary, 2 - HesAff binary.       - " << std::endl;
    std::cout << " -     For types 0 and 2, the program assumes that the descriptor entries are stored as uint8 - " << std::endl;
    std::cout << " -     while floats were used for type 1.                                                     - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  pos_db                                                                                    - " << std::endl;
    std::cout << " -     Text files containing the positions of the database images. Each line specifies the    - " << std::endl;
    std::cout << " -     two coordinates of one database image. Needs to be in the same order as list_db.       - " << std::endl;
    std::cout << " -     Assumed to be in a local coordinate system in which the Euclidean distance makes sense.- " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << "________________________________________________________________________________________________" << std::endl;
    return -1;
  }
  
  // Loads the inverted index.
  std::cout << " Loading the inverted index " << std::endl;
  geometric_burstiness::InvertedIndex<64> inverted_index;
  std::string index_file(argv[3]);
  if (!inverted_index.LoadInvertedIndex(index_file)) {
    std::cerr << "ERROR: Cannot load the inverted index from "
              << index_file << std::endl;
    return -1;
  }
  std::cout << " Index loaded " << std::endl;
  
  {
    std::string weight_file(index_file);
    weight_file.append(".weights");
    if (!inverted_index.ReadWeightsAndConstants(weight_file)) {
      std::cerr << "ERROR: Cannot load the weights and constants for the "
                << " inverted file from " << weight_file << std::endl;
      return -1;
    }
  }
  std::cout << " Weights loaded " << std::endl;

  std::vector<std::string> query_list;
  {
    std::ifstream ifs(argv[1], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot load the query list from " << argv[1]
                << std::endl;
      return -1;
    }

    while (!ifs.eof()) {
      std::string s = "";
      ifs >> s;
      if (!s.empty()) {
        if (query_list.empty()) {
          query_list.push_back(s);
        } else if (query_list.back().compare(s) != 0) {
          query_list.push_back(s);
        }
      }
    }

    ifs.close();
  }
  int num_query = static_cast<int>(query_list.size());
  std::cout << " Found " << num_query << " query files " << std::endl;

  // Loads the filenames of the database images.
  std::vector<std::string> db_list;
  {
    std::ifstream ifs(argv[2], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot load the db list from " << argv[2]
                << std::endl;
      return -1;
    }

    while (!ifs.eof()) {
      std::string s = "";
      ifs >> s;
      if (!s.empty()) {
        if (db_list.empty()) {
          db_list.push_back(s);
        } else if (db_list.back().compare(s) != 0) {
          db_list.push_back(s);
        }
      }
    }

    ifs.close();
  }
  int num_db = static_cast<int>(db_list.size());
  std::cout << " Found " << num_db << " db images in " << argv[2] << std::endl;

  // Loads the 2D positions for the database images.
  Eigen::Matrix2Xd db_pos(2, num_db);
  {
    std::ifstream ifs(argv[6], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot load the db positions from " << argv[9]
                << std::endl;
      return -1;
    }

    for (int i = 0; i < num_db; ++i) {
      double x = 0.0;
      double y = 0.0;
      ifs >> x >> y;
      db_pos(0, i) = x;
      db_pos(1, i) = y;
    }
    ifs.close();
  }

  ////
  // Opens the files for output.
  std::string output_prefix(argv[4]);

  std::string output_image_ranking(output_prefix);
  output_image_ranking.append(".retrieval_ranking.txt");
  std::ofstream ofs_retrieval(output_image_ranking.c_str(), std::ios::out);
  if (!ofs_retrieval.is_open()) {
    std::cerr << " ERROR: Cannot write the retrieval scores to "
              << output_image_ranking << std::endl;
    return -1;
  }

  std::string output_inlier_count_ranking(output_prefix);
  output_inlier_count_ranking.append(".inlier_count_ranking.txt");
  std::ofstream ofs_inlier_count(output_inlier_count_ranking.c_str(),
                                 std::ios::out);
  if (!ofs_inlier_count.is_open()) {
    std::cerr << " ERROR: Cannot write the inlier count scores to "
              << output_inlier_count_ranking << std::endl;
    return -1;
  }

  std::string output_eff_inlier_count_ranking(output_prefix);
  output_eff_inlier_count_ranking.append(".effective_inlier_count_ranking.txt");
  std::ofstream ofs_eff_inlier_count(output_eff_inlier_count_ranking.c_str(),
                                     std::ios::out);
  if (!ofs_eff_inlier_count.is_open()) {
    std::cerr << " ERROR: Cannot write the effective inlier count scores to "
              << output_eff_inlier_count_ranking << std::endl;
    return -1;
  }

  std::string output_inter_image_ranking(output_prefix);
  output_inter_image_ranking.append(
      ".inter_image_geometric_burstiness_ranking.txt");
  std::ofstream ofs_inter_image(output_inter_image_ranking.c_str(),
                                std::ios::out);
  if (!ofs_inter_image.is_open()) {
    std::cerr << " ERROR: Cannot write the inter-image burstiness scores to "
              << output_inter_image_ranking << std::endl;
    return -1;
  }

  std::string output_inter_place_ranking(output_prefix);
  output_inter_place_ranking.append(
      ".inter_place_geometric_burstiness_ranking.txt");
  std::ofstream ofs_inter_place(output_inter_place_ranking.c_str(),
                                std::ios::out);
  if (!ofs_inter_place.is_open()) {
    std::cerr << " ERROR: Cannot write the inter-place burstiness scores to "
              << output_inter_place_ranking << std::endl;
    return -1;
  }

  std::string output_inter_place_pop_ranking(output_prefix);
  output_inter_place_pop_ranking.append(
      ".inter_place_popularity_geometric_burstiness_ranking.txt");
  std::ofstream ofs_inter_place_pop(output_inter_place_pop_ranking.c_str(),
                                    std::ios::out);
  if (!ofs_inter_place_pop.is_open()) {
    std::cerr << " ERROR: Cannot write the inter-place+pop burstiness scores to "
              << output_inter_place_pop_ranking << std::endl;
    return -1;
  }

  ////
  // Performs the actual query.
  const int sift_type = atoi(argv[5]);
  bool invert_affine_matrix_when_loading_features = sift_type == 0;
  std::vector<int> correct_at_k(11, 0);
  std::vector<int> correct_at_k_uniq(11, 0);
  std::vector<int> correct_at_k_sp(11, 0);
  std::vector<int> correct_at_k_sp_uniq(11, 0);

  for (int i = 0; i < num_query; ++i) {
    std::cout << std::endl << " Query " << i << " : " << query_list[i]
              << std::endl;
    std::string sift_file(query_list[i]);
    sift_file.append("bin");

    std::vector<geometric_burstiness::AffineSIFTKeypoint> keypoints;
    std::vector<std::vector<uint32_t> > words;
    Eigen::Matrix<float, 128, Eigen::Dynamic> descriptors_float;
    Eigen::Matrix<uint8_t, 128, Eigen::Dynamic> descriptors_uint8_t;
    
    bool loaded = false;
    if (sift_type == 0 || sift_type == 2) {
      loaded = geometric_burstiness::LoadAffineSIFTFeaturesAndMultipleNNWordAssignments<uint8_t>(
        sift_file, invert_affine_matrix_when_loading_features, kNumNNWords,
        &keypoints, &words, &descriptors_uint8_t);
    } else {
      loaded = geometric_burstiness::LoadAffineSIFTFeaturesAndMultipleNNWordAssignments<float>(
        sift_file, invert_affine_matrix_when_loading_features, kNumNNWords,
        &keypoints, &words, &descriptors_float);
    }
    if (!loaded) {
      std::cerr << " ERROR: Cannot load the descriptors for query image " << i
          << " from " << sift_file << std::endl;
      return -1;
    }
    int num_features = static_cast<int>(keypoints.size());

    std::vector<geometric_burstiness::QueryDescriptor<64> > query_descriptors(
        num_features);

    for (int j = 0; j < num_features; ++j) {
      Eigen::Matrix<float, 128, 1> sift;
      if (sift_type == 0 || sift_type == 2) {
        sift = descriptors_uint8_t.col(j).cast<float>();
      } else {
        sift = descriptors_float.col(j);
      }

      inverted_index.PrepareQueryDescriptor(sift, &(query_descriptors[j]));
      for (int k = 0; k < kNumNNWords; ++k) {
        int w = static_cast<int>(words[j][k]);
        query_descriptors[j].relevant_word_ids.push_back(w);
      }
      if (j == (num_features -1)) std::cout << std::endl;
      query_descriptors[j].x = keypoints[j].x[0];
      query_descriptors[j].y = keypoints[j].x[1];
      query_descriptors[j].a = keypoints[j].a;
      query_descriptors[j].b = keypoints[j].b;
      query_descriptors[j].c = keypoints[j].c;
      query_descriptors[j].feature_id = j;
    }
    
    std::cout << "  Prepared " << query_descriptors.size() << " query descriptors " << std::endl;

    {
      for (int j = 0; j < num_features; ++j) {
        for (int w : query_descriptors[j].relevant_word_ids) {
          query_descriptors[j].max_hamming_distance_per_word.push_back(32);
        }
      }
    }

    std::vector<geometric_burstiness::ImageScore> image_scores;

    inverted_index.QueryIndex(query_descriptors, &image_scores);
    std::cout << "  Query found " << image_scores.size()
              << " potentially relevant database images" << std::endl;

    if (image_scores.empty()) {
      continue;
    }

    std::cout << "  Score of most relevant image "
              << image_scores[0].voting_weight << " (image "
              << image_scores[0].image_id << ") with "
              << image_scores[0].matches.size() << " matches" << std::endl;

    // Performs spatial verification for the top-kNumSpatial ranked database
    // images.
    std::cout << "  Starting spatial verification" << std::endl;
    int num_images_to_verify = std::min(kNumSpatial,
                                        static_cast<int>(image_scores.size()));
    // Data structures to store the various information required for re-ranking
    // based on the number of inliers, the effective inlier count, and the
    // geometric burstiness schemes.
    std::vector<std::pair<int, int> > db_num_inlier_pairs;
    std::vector<std::pair<double, int> > db_effective_inlier_count_pairs;
    std::vector<std::pair<int ,std::vector<int> > > db_inlier_ids_pairs;

    for (int r = 0; r < num_images_to_verify; ++r) {
      ofs_retrieval << query_list[i] << "jpg "
                    << db_list[image_scores[r].image_id] << " "
                    << image_scores[r].voting_weight << std::endl;
      if (image_scores[r].matches.size() < 4u) continue;
      using geometric_burstiness::geometry::FastSpatialMatching;
      using geometric_burstiness::geometry::FeatureGeometryAffine;
      using geometric_burstiness::geometry::Similarity5DOF;
      using geometric_burstiness::geometry::AffineFeatureMatch;
      using geometric_burstiness::geometry::AffineFeatureMatches;
      using geometric_burstiness::geometry::Transformation;
      AffineFeatureMatches matches_sfm;
      int num_matches = inverted_index.GetMatchesForSpatialVerification(
          query_descriptors, image_scores[r], &matches_sfm);
      if (num_matches < 4) continue;

      FastSpatialMatching<AffineFeatureMatch, FeatureGeometryAffine, Similarity5DOF> verifier;

      std::vector<std::pair<int, int> > inlier_ids;
      Transformation transform;
      int num_inliers = verifier.PerformSpatialVerification(matches_sfm,
                                                            &transform,
                                                            &inlier_ids);
      if (num_inliers >= 4) {
        int db_idx = image_scores[r].image_id;
        db_num_inlier_pairs.push_back(std::make_pair(num_inliers, db_idx));

        // Computes information about the effective inlier count.
        double eff_inlier_count =
            geometric_burstiness::geometry::ComputeEffectiveInlierCount(
                matches_sfm, inlier_ids);
        db_effective_inlier_count_pairs.push_back(
            std::make_pair(eff_inlier_count, db_idx));

        // Computes the information required for handling geometric bursts.
        std::pair<int, std::vector<int> > geom_burst_info;
        geom_burst_info.first = db_idx;
        geom_burst_info.second.clear();
        for (const std::pair<int, int>& in_ids : inlier_ids) {
          int q_feat_id = matches_sfm[in_ids.first].feature1_.feature_id_;
          geom_burst_info.second.push_back(q_feat_id);
        }
        db_inlier_ids_pairs.push_back(geom_burst_info);
      }
    }

    // Computes the rankings for the different strategies.
    std::vector<std::pair<int, int> > ranking_inlier_count;
    geometric_burstiness::reranking::ReRankingInlierCount(
        db_num_inlier_pairs, &ranking_inlier_count);

    std::vector<std::pair<double, int> > ranking_effective_inlier_count;
    geometric_burstiness::reranking::ReRankingEffectiveInlierCount(
        db_effective_inlier_count_pairs, &ranking_effective_inlier_count);

    std::vector<std::pair<double, int> > ranking_inter_image_burstiness;
    geometric_burstiness::reranking::ReRankingInterImageGeometricBurstiness(
        db_inlier_ids_pairs, num_features, &ranking_inter_image_burstiness);

    // For inter-place burstiness ranking, the list of retrieved images needs
    // to be in descending order of inliers. This is achieved by sorting.
    // We use stable sort to preserve the initial ranking if required.
    std::stable_sort(db_inlier_ids_pairs.begin(), db_inlier_ids_pairs.end(),
                     CompareBasedOnNumberOfInliers<int, int>);

    std::vector<std::pair<double, int> > ranking_inter_place_burstiness;
    std::vector<std::pair<double, int> > ranking_inter_place_pop_burstiness;
    geometric_burstiness::reranking::ReRankingInterPlaceGeometricBurstiness(
        db_inlier_ids_pairs, num_features, db_pos, squared_gt_inlier_dist,
        &ranking_inter_place_burstiness, &ranking_inter_place_pop_burstiness);

    if (!ranking_inlier_count.empty()) {
      std::cout << "  Most relevant image according to " << std::endl
                << "    inlier count: " << ranking_inlier_count[0].second
                << std::endl << "    effective inlier count: "
                << ranking_effective_inlier_count[0].second << std::endl
                << "    inter-image burstiness: "
                << ranking_inter_image_burstiness[0].second << std::endl
                << "    inter-place burstiness: "
                << ranking_inter_place_burstiness[0].second << std::endl
                << "    inter-place + popularity burstiness: "
                << ranking_inter_place_pop_burstiness[0].second << std::endl;
    }

    for (size_t r = 0; r < ranking_inlier_count.size(); ++r) {
      ofs_inlier_count << query_list[i] << "jpg "
                       << db_list[ranking_inlier_count[r].second] << " "
                       << ranking_inlier_count[r].first << std::endl;
    }

    for (size_t r = 0; r < ranking_effective_inlier_count.size(); ++r) {
      ofs_eff_inlier_count << query_list[i] << "jpg "
                           << db_list[ranking_effective_inlier_count[r].second]
                           << " " << ranking_effective_inlier_count[r].first
                           << std::endl;
    }

    for (size_t r = 0; r < ranking_inter_image_burstiness.size(); ++r) {
      ofs_inter_image << query_list[i] << "jpg "
                      << db_list[ranking_inter_image_burstiness[r].second]
                      << " " << ranking_inter_image_burstiness[r].first
                      << std::endl;
    }

    for (size_t r = 0; r < ranking_inter_place_burstiness.size(); ++r) {
      ofs_inter_place << query_list[i] << "jpg "
                      << db_list[ranking_inter_place_burstiness[r].second]
                      << " " << ranking_inter_place_burstiness[r].first
                      << std::endl;
    }

    for (size_t r = 0; r < ranking_inter_place_pop_burstiness.size(); ++r) {
      ofs_inter_place_pop << query_list[i] << "jpg "
                          << db_list[ranking_inter_place_pop_burstiness[r].second]
                          << " " << ranking_inter_place_pop_burstiness[r].first
                          << std::endl;
    }
  }

  ofs_retrieval.close();
  ofs_inlier_count.close();
  ofs_eff_inlier_count.close();
  ofs_inter_image.close();
  ofs_inter_place.close();
  ofs_inter_place_pop.close();

  return 0;
}

