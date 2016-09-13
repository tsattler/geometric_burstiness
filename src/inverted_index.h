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

#ifndef GEOMETRIC_BURSTINESS_SRC_INVERTED_INDEX_H_
#define GEOMETRIC_BURSTINESS_SRC_INVERTED_INDEX_H_

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "fast_spatial_matching.h"
#include "geometric_transforms.h"
#include "structures.h"

namespace geometric_burstiness {
// Skip every inverted file with less than kMinEntries entries.
const size_t kMinEntries = 10u;

// Implements an inverted index system. The template parameter is the length of
// the binary vectors obtained via Hamming Embedding.
template <int N>
class InvertedIndex {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef Eigen::Matrix<float, N, 1> ProjectedDesc;

  InvertedIndex() : num_visual_words_(-1) {
    projection_matrix_.setIdentity();
  }

  void SetProjectionMatrix(
      const Eigen::Matrix<float, N, 128, Eigen::RowMajor>& projection) {
    projection_matrix_ = projection;
  }

  // Initializes the inverted index with num_words empty inverted files.
  void InitializeIndex(int num_words) {
    num_visual_words_ = num_words;
    if (num_visual_words_ <= 0) {
      inverted_files_.clear();
      num_visual_words_ = 0;
      inverted_file_initialized_.clear();
    } else {
      inverted_files_.resize(num_visual_words_);
      inverted_file_initialized_.resize(num_words);
      for (int i = 0; i < num_words; ++i) {
        inverted_files_[i].ClearFile();
        inverted_files_[i].SetWordId(i);
        inverted_file_initialized_[i] = false;
      }
    }
  }

  // Learns the parameters for Hamming embedding from a set of descriptors with
  // given visual word ids.
  void LearnHammingEmbeddingParameters(
      const Eigen::Matrix<uint8_t, 128, Eigen::Dynamic>& descriptors,
      int num_desc,
      const std::vector<int>& word_ids) {
    if (num_desc > descriptors.cols()) {
      std::cerr << "ERROR: Number of descriptors too large correct: "
                << num_desc << " " << descriptors.cols() << std::endl;
      return;
    }
    if (static_cast<int>(word_ids.size()) != num_desc) {
      std::cerr << "ERROR: The number of words does not match the number of "
                << "descriptors" << std::endl;
      return;
    }
    
    // Determines for each word the corresponding descriptors.
    std::vector<std::vector<int> > indices_per_word(num_visual_words_);
    for (int j = 0; j < num_desc; ++j) {
      indices_per_word[word_ids[j]].push_back(j);
    }

    // For each word, learn the Hamming embedding threshold and the local
    // descriptor space densities.
    int num_skipped_words = 0;
    for (int i = 0; i < num_visual_words_; ++i) {
      if (indices_per_word[i].size() < kMinEntries) {
        ++num_skipped_words;
        continue;
      }

      size_t num_entries = indices_per_word[i].size();
      Eigen::Matrix<float, N, Eigen::Dynamic> proj_desc(N, num_entries);
      for (size_t j = 0; j < num_entries; ++j) {
        proj_desc.col(j) = projection_matrix_
            * descriptors.col(indices_per_word[i][j]).cast<float>();
      }

      inverted_file_initialized_[i] =
        inverted_files_[i].LearnEmbeddingThreshold(proj_desc);

      if (i % 1000 == 0) {
        std::cout << "\r Learning Hamming embedding thresholds " << i
                  << std::flush;
      }
    }
    std::cout << "   skipped " << num_skipped_words << " words " << std::endl;
  }

  // For each word, learn the Hamming embedding threshold and the local
  // descriptor space densities.
  void EstimateDescriptorSpaceDensities() {
    HammingSpace<8> hamming_space;

    for (int i = 0; i < num_visual_words_; ++i) {
      if (!inverted_file_initialized_[i]) continue;
      inverted_files_[i].LearnSigmas(hamming_space);
      if (i % 1000 == 0) std::cout << "\r " << i << std::flush;
    }
    std::cout << std::endl;
  }


//  // Given a set of inverted file entries and their descriptors, adds them to
//  // the inverted file. This function assumes that the thresholds for Hamming
//  // embedding have already been learned.
//  void AddEntries(
//      const std::vector<InvertedFileEntry<N> >& entries,
//      const std::vector<int>& word_ids,
//      const Eigen::Matrix<uint8_t, 128, Eigen::Dynamic>& descriptors) {
//    assert(
//        static_cast<int>(entries.size())
//            == static_cast<int>(descriptors.cols()));
//    assert(entries.size() == word_ids.size());
//
//    int num_entries = static_cast<int>(entries.size());
//    for (int i = 0; i < num_entries; ++i) {
//      if (!inverted_file_initialized_[word_ids[i]]) continue;
//      AddEntry(entries[i], word_ids[i], descriptors.col(i));
//    }
//  }

  void AddEntry(const InvertedFileEntry<N>& entry, int word_id,
                const Eigen::Matrix<float, 128, 1>& descriptor) {
    ProjectedDesc proj_desc = projection_matrix_ * descriptor.cast<float>();
    inverted_files_[word_id].AddEntry(proj_desc, entry);
  }

  // Prepares a query descriptor by projecting the full SIFT descriptor into
  // the lower dimensional space.
  void PrepareQueryDescriptor(const Eigen::Matrix<float, 128, 1>& sift,
                              QueryDescriptor<N>* query_desc) const {
    query_desc->proj_desc = projection_matrix_ * sift;
  }

  // Returns the binary representation for a given projected descriptor and its
  // visual word.
  void GetBinaryDescriptor(const Eigen::Matrix<float, N, 1>& desc, int word,
                           std::bitset<N>* b) const {
    inverted_files_[word].ConvertToBinaryString(desc, b);
  }

  // Finalizes the inverted index by sorting each inverted file such that all
  // entries are in ascending order of image ids.
  void FinalizeIndex() {
    for (int i = 0; i < num_visual_words_; ++i) {
      if (!inverted_file_initialized_[i]) continue;
      inverted_files_[i].SortInvertedFile();
    }
  }

  // Queries the inverted file and returns a list of votes (together with 2D-2D
  // matches for these images) for database images. This list is sorted in
  // descending order of relevance to the query. DOES NOT perform spatial
  // verification.
  void QueryIndex(const std::vector<QueryDescriptor<N> >& query_descriptors,
                  std::vector<ImageScore>* image_scores) const {
    image_scores->clear();

    // Computes the self-similarity score for the query image.
    int num_query = static_cast<int>(query_descriptors.size());
    double self_similarity = ComputeQueryImageSelfSimilarity(query_descriptors);
    float normalization_weight = 1.0f;
    if (self_similarity > 0.0f) {
      normalization_weight = static_cast<float>(1.0 / sqrt(self_similarity));
    }
    std::cout << "  Self-similarity: " << self_similarity << std::endl;
    std::cout << "  Normalization weight " << normalization_weight << std::endl;


    std::unordered_map<int, int> score_map;
    std::unordered_map<int, int>::iterator map_it;

    for (int i = 0; i < num_query; ++i) {
      for (const int w_id : query_descriptors[i].relevant_word_ids) {
        if (!inverted_file_initialized_[w_id]) continue;
        
        std::vector<ImageScore> query_scores;

        inverted_files_[w_id].PerformVoting(query_descriptors[i],
                                            vote_weighting_function_,
                                            idf_weights_[w_id], &query_scores);

        for (ImageScore& score : query_scores) {
          score.voting_weight *= idf_weights_[w_id] * idf_weights_[w_id];
          map_it = score_map.find(score.image_id);
          if (map_it == score_map.end()) {
            score_map[score.image_id] = static_cast<int>(image_scores->size());
            image_scores->push_back(score);
          } else {
            (*image_scores)[map_it->second].voting_weight += score.voting_weight;
            (*image_scores)[map_it->second].matches.insert(
                (*image_scores)[map_it->second].matches.end(),
                score.matches.begin(), score.matches.end());
          }
        }
      }
    }

    // Normalization.
    for (ImageScore& score : *image_scores) {
      float c1 = normalization_constants_[score.image_id];
//      std::cout << "  " << score.image_id << " " << score.matches.size()
//                << " " << score.voting_weight << " " << score.voting_weight * normalization_weight * c1 << std::endl;
      score.voting_weight *= (normalization_weight * c1);
    }

    std::sort(image_scores->begin(), image_scores->end(), CmpImageScores);
  }

  // Given a query image represented as a set of query descriptors, computes the
  // self-similarity for the image.
  double ComputeQueryImageSelfSimilarity(
      const std::vector<QueryDescriptor<N> >& query_descriptors) const {
    double self_similarity = 0.0;

    int num_query = static_cast<int>(query_descriptors.size());
    typedef std::pair<int, std::bitset<N> > WordDescPair;
    std::vector<std::vector<WordDescPair> > bin_descriptors_per_query(
        num_query);

    std::cout << "  Computing the self-similarity for the query image from "
              << num_query << " query descriptors " << std::endl;
    for (int i = 0; i < num_query; ++i) {
      bin_descriptors_per_query[i].clear();
      for (const int w_id : query_descriptors[i].relevant_word_ids) {
        if (!inverted_file_initialized_[w_id]) continue;

        std::bitset<N> b;
        inverted_files_[w_id].ConvertToBinaryString(
            query_descriptors[i].proj_desc, &b);
        bin_descriptors_per_query[i].push_back(std::make_pair(w_id, b));
      }
    }

    for (int i = 0; i < num_query; ++i) {
      for (const WordDescPair& pi : bin_descriptors_per_query[i]) {
        if (!inverted_file_initialized_[pi.first]) continue;
        double idf_squared = static_cast<double>(idf_weights_[pi.first])
            * static_cast<double>(idf_weights_[pi.first]);
        self_similarity += idf_squared;
      }
    }
    return self_similarity;
  }

  // Given a set of matches for a database images, where each match is given as
  // indices, returns a set of correspondences suitable for spatial
  // verification. Returns the number of matches.
  int GetMatchesForSpatialVerification(
      const std::vector<QueryDescriptor<N> >& query_descriptors,
      const ImageScore& initial_matches,
      geometry::AffineFeatureMatches* matches_fsm) const {
    matches_fsm->clear();

    if (initial_matches.matches.size() < 4u) return 0;
    // Since each feature in a query image might match to multiple database
    // image features, we need to keep track on which features are already used.
    std::unordered_map<int, size_t> feature_map;
    std::unordered_map<int, size_t>::iterator map_it;

    // Similarly, we need to assign a unique id to each database feature.
    std::unordered_map<std::string, int> db_feature_ids;
    std::unordered_map<std::string, int>::iterator db_f_ids_it;
    int current_db_feature_index = 0;

    int num_matches = 0;
    for (const Match2D2D& init_match: initial_matches.matches) {
      int query_feature_id = init_match.query_feature_id;
      map_it = feature_map.find(query_feature_id);

      const InvertedFileEntry<N>& entry = inverted_files_[init_match
          .db_feature_word].GetIthEntry(init_match.db_feature_index);
      std::string word_id_string;
      WordFeatureIndexPairToString(init_match.db_feature_word,
                                   init_match.db_feature_index,
                                   &word_id_string);
      db_f_ids_it = db_feature_ids.find(word_id_string);
      int used_db_feature_id = current_db_feature_index;
      if (db_f_ids_it != db_feature_ids.end()) {
        used_db_feature_id = db_f_ids_it->second;
      } else {
        db_feature_ids.insert(
            std::make_pair(word_id_string, current_db_feature_index));
        ++current_db_feature_index;
      }

      if (map_it != feature_map.end()) {
        geometry::AffineFeatureMatch& match = (*matches_fsm)[map_it->second];

        geometry::FeatureGeometryAffine f;

        f.feature_id_ = used_db_feature_id;
        f.x_ << entry.x, entry.y;
        f.a_ = entry.a;
        f.b_ = entry.b;
        f.c_ = entry.c;
//        f.scale_ = entry.scale;
//        f.orientation_ = entry.orientation;
        match.features2_.push_back(f);
        match.word_ids_.push_back(init_match.db_feature_word);
      } else {
        feature_map[query_feature_id] = matches_fsm->size();
        geometry::AffineFeatureMatch match;
        match.feature1_.feature_id_ = query_feature_id;
        match.feature1_.x_ << query_descriptors[query_feature_id].x,
                              query_descriptors[query_feature_id].y;
        match.feature1_.a_ = query_descriptors[query_feature_id].a;
        match.feature1_.b_ = query_descriptors[query_feature_id].b;
        match.feature1_.c_ = query_descriptors[query_feature_id].c;
//        match.feature1_.scale_ = query_descriptors[query_feature_id].scale;
//        match.feature1_.orientation_ = query_descriptors[query_feature_id]
//            .orientation;

        geometry::FeatureGeometryAffine f;
        f.feature_id_ = used_db_feature_id;
        f.x_ << entry.x, entry.y;
        f.a_ = entry.a;
        f.b_ = entry.b;
        f.c_ = entry.c;
//        f.scale_ = entry.scale;
//        f.orientation_ = entry.orientation;
        match.features2_.push_back(f);
        match.word_ids_.push_back(init_match.db_feature_word);
        matches_fsm->push_back(match);
      }
      ++num_matches;
    }
    std::sort(matches_fsm->begin(), matches_fsm->end(),
              geometry::CmpFeature<geometry::FeatureGeometryAffine>);

    return num_matches;
  }

  // Given a set of matches for a database images, where each match is given as
  // indices, returns a set of 1-to-1 correspondences suitable for spatial
  // verification. To select 1-to-1 matches, the approach from Li et al.,
  // Pairwise Geometric Matching for Large-scale Object Retrieval. CVPR 2015
  // is used (cf. Sec. 4.1 in the paper). Returns the number of matches.
  int Get1To1MatchesForSpatialVerification(
      const std::vector<QueryDescriptor<N> >& query_descriptors,
      const ImageScore& initial_matches,
      geometry::AffineFeatureMatches* matches_fsm) const {
    matches_fsm->clear();

    if (initial_matches.matches.size() < 4u) return 0;
    typedef std::pair<int, float> WeightedMatch;
    typedef std::vector<WeightedMatch> WeightedMatchVec;

    // Stores for each query feature a vector of relevant database features.
    // Stores for each database feature a vector  of relevant query features and
    // assigns it a unique id. Also, stores for each database feature its
    // affine parameters.
    std::unordered_map<int, WeightedMatchVec> matches_per_query;
    std::unordered_map<std::string, int> database_ids;
    std::unordered_map<int, WeightedMatchVec> matches_per_db;
    std::unordered_map<int, geometry::FeatureGeometryAffine> db_feat_geometries;
    std::unordered_map<int, int> db_words;

    int current_db_feature_index = 0;
    for (const Match2D2D& init_match : initial_matches.matches) {
      int query_feature_id = init_match.query_feature_id;
      // Get the id of the database feature.
      std::string word_id_string;
      WordFeatureIndexPairToString(init_match.db_feature_word,
                                   init_match.db_feature_index,
                                   &word_id_string);
      int db_feat_id = database_ids.emplace(word_id_string,
                                            current_db_feature_index).first
          ->second;

      const InvertedFileEntry<N>& entry = inverted_files_[init_match
          .db_feature_word].GetIthEntry(init_match.db_feature_index);

      db_feat_geometries[db_feat_id].feature_id_ = db_feat_id;
      db_feat_geometries[db_feat_id].x_ << entry.x, entry.y;
      db_feat_geometries[db_feat_id].a_ = entry.a;
      db_feat_geometries[db_feat_id].b_ = entry.b;
      db_feat_geometries[db_feat_id].c_ = entry.c;
      db_words[db_feat_id] = init_match.db_feature_word;

      // Updates the matches found for the query and database images.
      matches_per_query[init_match.query_feature_id].push_back(
          std::make_pair(db_feat_id, init_match.weight));

      matches_per_db[db_feat_id].push_back(
          std::make_pair(init_match.query_feature_id, init_match.weight));

      ++current_db_feature_index;
    }

    // Stores for each feature in each image the number of possible matches from
    // which we can choose. At the same time, populates two vectors containing
    // a pair (feature_id, num_matches) for each feature.
    std::unordered_map<int, int> num_matches_per_query;
    std::unordered_map<int, bool> query_feature_selected;

    std::unordered_map<int, int> num_matches_per_db;
    std::unordered_map<int, bool> db_feature_selected;

    for (const std::pair<int, WeightedMatchVec>& m : matches_per_query) {
      int num_potential_matches = static_cast<int>(m.second.size());
      num_matches_per_query[m.first] = num_potential_matches;
      query_feature_selected[m.first] = false;
    }
    for (const std::pair<int, WeightedMatchVec>& m : matches_per_db) {
      int num_potential_matches = static_cast<int>(m.second.size());
      num_matches_per_db[m.first] = num_potential_matches;
      db_feature_selected[m.first] = false;
    }

    // Actually selects the matches.
    // TODO(sattler): Think about a faster implementation!
    int num_matches = 0;
    int best_query_id = -1;
    int best_query_value = -1;
    int best_db_id = -1;
    int best_db_value = -1;
    while (true) {
      // Obtains the two features (one in the query image, the other in the
      // database image) for which the smallest number of potential matches
      // exists and selects the one contained in the lowest number of matches.
      GetIdAndValueForLowestValue(num_matches_per_query, &best_query_id, &best_query_value);
      GetIdAndValueForLowestValue(num_matches_per_db, &best_db_id, &best_db_value);

      if (best_query_id == -1 && best_db_id == -1) {
        // No more matches can be found.
        break;
      }

      bool use_query = best_query_value < best_db_value;
      if (best_query_value == -1) use_query = false;

      // Finds the best match with the highest similarity score.
      int best_matching_feature = -1;
      float best_matching_score = -1.0f;
      const WeightedMatchVec& vec =
          use_query ?
              matches_per_query[best_query_id] : matches_per_db[best_db_id];
      const std::unordered_map<int, bool>& feat_selected =
          use_query ? db_feature_selected : query_feature_selected;
      for (const WeightedMatch& m : vec) {
        if ((m.second > best_matching_score)
            && (feat_selected.at(m.first) == false)) {
          best_matching_score = m.second;
          best_matching_feature = m.first;
        }
      }

      // Adds the match.
      ++num_matches;
      geometry::AffineFeatureMatch new_match;
      const int f1_id = use_query? best_query_id : best_matching_feature;
      new_match.feature1_.feature_id_ = f1_id;
      new_match.feature1_.x_ << query_descriptors[f1_id].x,
                                query_descriptors[f1_id].y;
      new_match.feature1_.a_ = query_descriptors[f1_id].a;
      new_match.feature1_.b_ = query_descriptors[f1_id].b;
      new_match.feature1_.c_ = query_descriptors[f1_id].c;

      const int f2_id = use_query? best_matching_feature : best_db_id;
      new_match.features2_.push_back(db_feat_geometries[f2_id]);
      new_match.word_ids_.push_back(db_words[f2_id]);
      matches_fsm->push_back(new_match);

//      std::cout << " selected: " << use_query << " " << f1_id << " " << f2_id << std::endl;
//      std::cout << db_feat_geometries[f2_id].x_.transpose() << " " << db_feat_geometries[f2_id].a_ << " " << db_feat_geometries[f2_id].b_ << " " << db_feat_geometries[f2_id].c_ << std::endl;

      // Updates the list of potential matches.
      query_feature_selected[f1_id] = true;
      db_feature_selected[f2_id] = true;
      num_matches_per_query.erase(f1_id);
      num_matches_per_db.erase(f2_id);
      for (const WeightedMatch& m : matches_per_query[f1_id]) {
        std::unordered_map<int, int>::iterator it = num_matches_per_db.find(
            m.first);
        if (it != num_matches_per_db.end()) {
          it->second -= 1;
          if (it->second <= 0) num_matches_per_db.erase(it->first);
        }
      }
      for (const WeightedMatch& m : matches_per_db[f2_id]) {
        std::unordered_map<int, int>::iterator it = num_matches_per_query.find(
            m.first);
        if (it != num_matches_per_query.end()) {
          it->second -= 1;
          if (it->second <= 0) num_matches_per_query.erase(it->first);
        }
      }
    }

    return num_matches;
  }

  // Writes the inverted index into a binary file of the given name. Returns
  // false if the inverted index cannot be written to disk.
  bool SaveInvertedIndex(const std::string& filename) const {
    std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
      std::cerr << "ERROR: Cannot write the inverted index to " << filename
                << std::endl;
      return false;
    }

    // First writes out the number of visual words.
    int32_t num_words = static_cast<int32_t>(num_visual_words_);
    ofs.write((const char*) &num_words, sizeof(int32_t));

    // Writes out the length of the binary strings.
    int32_t N_t = static_cast<int32_t>(N);
    ofs.write((const char*) &N_t, sizeof(int32_t));

    // Writes out the projection matrix.
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 128; ++j) {
        float val = projection_matrix_(i, j);
        ofs.write((const char*) &val, sizeof(float));
      }
    }

    // Next, writes out the inverted files.
    for (int i = 0; i < num_visual_words_; ++i) {
      if (!inverted_files_[i].SaveInvertedFile(&ofs)) {
        ofs.close();
        return false;
      }
    }
    
    // Writes out which inverted files were initialized.
    for (int i = 0; i < num_visual_words_; ++i) {
      uint8_t word_initialized = static_cast<uint8_t>(
        inverted_file_initialized_[i]? 1 : 0);
      ofs.write((const char*) &word_initialized, sizeof(uint8_t));
    }

    ofs.close();

    return true;
  }

  // Reads the inverted index from a binary file of the given name. Returns
  // false if the inverted index cannot be read from disk.
  bool LoadInvertedIndex(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot read the inverted index from " << filename
                << std::endl;
      return false;
    }

    int32_t num_words = 0;
    ifs.read((char*) &num_words, sizeof(int32_t));
    if (num_words <= 0) {
      std::cerr << "ERROR: The number of visual words in the file is not"
                << " positive." << std::endl;
      ifs.close();
      return false;
    }
    InitializeIndex(static_cast<int>(num_words));

    // Reads in the length of the binary strings.
    int32_t N_t = 0;
    ifs.read((char*) &N_t, sizeof(int32_t));
    if (static_cast<int>(N_t) != N) {
      std::cerr << "ERROR: The length of the binary strings should be " << N
                << " but is " << N_t << ". The indices are not compatible!"
                << std::endl;
      ifs.close();
      return false;
    }

    // Reads in the projection matrix.
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 128; ++j) {
        float val = 0.0f;
        ifs.read((char*) &val, sizeof(float));
        projection_matrix_(i, j) = val;
      }
    }

    // Next, reads all inverted files.
    for (int i = 0; i < num_visual_words_; ++i) {
      if (!inverted_files_[i].ReadInvertedFile(&ifs)) {
        ifs.close();
        std::cerr << "ERROR: Cannot read inverted file " << i << std::endl;
        return false;
      }
    }
    
    for (int i = 0; i < num_visual_words_; ++i) {
      uint8_t word_initialized = 0;
      ifs.read((char*) &word_initialized, sizeof(uint8_t));
      if (word_initialized == 1) {
        inverted_file_initialized_[i] = true;
      } else {
        inverted_file_initialized_[i] = false;
      }
    }

    ifs.close();

    return true;
  }
  
  void GetImageIdsInIndex(std::unordered_set<int>* ids) const {
    for (int i = 0; i < num_visual_words_; ++i) {
      inverted_files_[i].GetImageIdsInFile(ids);
    }
  }
  
  void GetEntriesForImage(int image_id,
                          std::vector<InvertedFileEntry<N> >* entries) const {
    entries->clear();
    for (int i = 0; i < num_visual_words_; ++i) {
      inverted_files_[i].GetEntriesForImage(image_id, entries);
    }
  }
  
  // Given a projected descriptor and the corresponding visual word, returns the
  // corresponding binary vector.
  void GetBinaryDescriptor(int word_id, const ProjectedDesc& desc,
                           std::bitset<N>* bin_desc) const {
    inverted_files_[word_id].ConvertToBinaryString(desc, bin_desc);
  }

  // Computes the idf-weights as well as the normalization constants for each
  // image. Assumes that the index has been finalized.
  bool ComputeWeightsAndNormalizationConstants() {
    if (num_visual_words_ <= 0) return false;
    if (inverted_files_.empty()) return false;

    // Determines the ids of all images in the database.
    std::cout << " Determining the ids of all images in the database"
              << std::endl;
    std::unordered_set<int> all_image_ids;
    for (int i = 0; i < num_visual_words_; ++i) {
      if (!inverted_file_initialized_[i]) continue;
      inverted_files_[i].GetImageIdsInFile(&all_image_ids);
    }
    int max_id = -1;
    for (const int& id : all_image_ids) {
      max_id = std::max(max_id, id);
    }
    std::cout << " Maximum image id: " << max_id << std::endl;
    std::cout << " The index contains " << all_image_ids.size() << " images"
              << std::endl;
    int total_num_images = static_cast<int>(all_image_ids.size());

    // Computes the idf weights.
    std::cout << " Computing the idf-weights " << std::endl;
    idf_weights_.resize(num_visual_words_);
    int num_skipped = 0;
    for (int i = 0; i < num_visual_words_; ++i) {
      idf_weights_[i] = 0.0f;
      if (!inverted_file_initialized_[i]) {
        ++num_skipped;
      } else {
        idf_weights_[i] = inverted_files_[i].GetIDFWeight(total_num_images);
      }
    }
    std::cout << "   skipped " << num_skipped << " words" << std::endl;

    // Computes the self-similarities of all database images to obtain the
    // normalization constants.
    std::cout << " Computing the normalization constants" << std::endl;
    normalization_constants_.resize(max_id + 1, 0.0);
    std::vector<double> self_similarities(max_id + 1, 0.0);
    for (int i = 0; i < num_visual_words_; ++i) {
      if (!inverted_file_initialized_[i]) continue;

      if (i != inverted_files_[i].WordId()) {
        std::cout << "ERROR: Expected word id " << i << " but found "
                  << inverted_files_[i].WordId() << std::endl;
      }

      if (!inverted_files_[i].DetermineDBImageSelfSimilarities(
          vote_weighting_function_, idf_weights_[i], &self_similarities)) {
        std::cerr << " ERROR: Could not compute the self-similarity for word "
                  << i << std::endl;
        return false;
      }
    }
    for (int i = 0; i <= max_id; ++i) {
      normalization_constants_[i] = 0.0f;
      if (self_similarities[i] > 0.0) {
        normalization_constants_[i] = static_cast<float>(1.0
            / sqrt(self_similarities[i]));
      }
    }

    return true;
  }

  // Writes the idf-weights and normalization constants to a binary file.
  bool SaveWeightsAndConstants(const std::string& filename) const {
    std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
      std::cerr << "ERROR: Cannot write the weights to " << filename
                << std::endl;
      return false;
    }

    // First writes out the number of visual words.
    int32_t num_words = static_cast<int32_t>(num_visual_words_);
    ofs.write((const char*) &num_words, sizeof(int32_t));

    // Writes out the idf-weights.
    for (int i = 0; i < num_visual_words_; ++i) {
      float w = idf_weights_[i];
      ofs.write((const char*) &w, sizeof(float));
    }

    // Writes out the number of images.
    uint32_t num_images = static_cast<uint32_t>(normalization_constants_.size());
    ofs.write((const char*) &num_images, sizeof(int32_t));

    // Writes out the normalization constants.
    for (uint32_t i = 0; i < num_images; ++i) {
      float w = normalization_constants_[i];
      ofs.write((const char*) &w, sizeof(float));
    }
    ofs.close();

    return true;
  }

  // Initializes the weights and constants from a binary file stored on disk.
  bool ReadWeightsAndConstants(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot read the weights from " << filename
                << std::endl;
      return false;
    }

    int32_t num_words = 0;
    ifs.read((char*) &num_words, sizeof(int32_t));
    if (num_words <= 0) {
      std::cerr << "ERROR: The number of visual words in the file is not"
                << " positive." << std::endl;
      ifs.close();
      return false;
    }
    idf_weights_.resize(num_words);
    for (uint32_t i = 0; i < num_words; ++i) {
      float w = 0.0f;
      ifs.read((char*) &w, sizeof(float));
      idf_weights_[i] = w;
    }

    int32_t num_images = 0;
    ifs.read((char*) &num_images, sizeof(int32_t));
    if (num_images <= 0) {
      std::cerr << "ERROR: The number of images in the index is not"
                << " positive." << std::endl;
      ifs.close();
      return false;
    }
    normalization_constants_.resize(num_images);
    for (uint32_t i = 0; i < num_images; ++i) {
      float w = 0.0f;
      ifs.read((char*) &w, sizeof(float));
      normalization_constants_[i] = w;
    }
    ifs.close();

    return true;
  }

  void PrintBinaryDescriptorStats() const {
    for (int i = 0; i < num_visual_words_; ++i) {
      inverted_files_[i].PrintBinaryDescriptorStats();
    }
  }

  bool SetHammingThresholds(
      const Eigen::Matrix<float, N, Eigen::Dynamic>& thresholds) {
    if (thresholds.cols() != num_visual_words_) return false;
    for (int i = 0; i < num_visual_words_; ++i) {
      inverted_files_[i].SetHammingThresholds(thresholds.col(i));
      inverted_file_initialized_[i] = true;
    }

    return true;
  }

  bool LoadSigmasRelja(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.is_open()) return false;
    int num_sigmas_per_subspace = 256;
    int num_subspaces = 8;
    std::vector<uint8_t> vals(256);
    double diffNorm = 0.0;
    uint32_t temp_ = 0;

    for (int word_id = 0; word_id < num_visual_words_; ++word_id) {
      Eigen::Matrix<float, 256, Eigen::Dynamic> local_sigmas(256, 8);
      for (int s = 0; s < num_subspaces; ++s) {
        for (int i = 0; i < num_sigmas_per_subspace; ++i) {
          uint8_t val = 0;
          ifs.read((char*) &val, sizeof(uint8_t));
          local_sigmas(i, s) = static_cast<float>(val);
        }
      }
      ifs.read((char*) &diffNorm, sizeof(double));
      float diff_norm_float = static_cast<float>(diffNorm);
      for (int s = 0; s < num_subspaces; ++s) {
        for (int i = 0; i < num_sigmas_per_subspace; ++i) {
          local_sigmas(i, s) = std::max(local_sigmas(i, s) + diff_norm_float,
                                        0.01f);
        }
      }

      inverted_files_[word_id].SetSigmasRelja(local_sigmas);
    }
    ifs.close();

    return true;
  }

 private:
  // Maps a word - feature index pair to a string that can be used for indexing.
  void WordFeatureIndexPairToString(int word_id, int feature_index,
                                    std::string* s) const {
    std::stringstream s_stream;
    s_stream << word_id << "_" << feature_index;
    *s = s_stream.str();
  }

  // Given an unordered map of (id, value) pairs, returns the id and value of
  // the pair with lowest value. id is -1 if no such element can be found.
  void GetIdAndValueForLowestValue(const std::unordered_map<int, int>& map,
                                   int* id, int *value) const {
    *id = -1;
    int val = std::numeric_limits<int>::max();
    for (const std::pair<int, int>& it : map) {
      if (it.second < val) {
        val = it.second;
        *id = it.first;
      }
    }

    *value = val;
  }

  // The idf weights.
  std::vector<float> idf_weights_;
  // The individual inverted indices.
  std::vector<InvertedFile<N> > inverted_files_;
  std::vector<bool> inverted_file_initialized_;
  // For each image in the database, a normalization factor to be used to
  // normalize the votes.
  std::vector<float> normalization_constants_;
  int num_visual_words_;
  // The projection matrix used to project SIFT descriptors.
  Eigen::Matrix<float, N, 128, Eigen::RowMajor> projection_matrix_;
  // The function used to weight the votes.
  VoteWeightingFunction<N> vote_weighting_function_;
};

}  // namespace geometric_burstiness

#endif  // GEOMETRIC_BURSTINESS_SRC_INVERTED_INDEX_H_
