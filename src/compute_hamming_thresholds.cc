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
#include <iostream>
#include <fstream>
#include <random>
#include <stdint.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "inverted_file.h"
#include "inverted_index.h"
#include "structures.h"


int main (int argc, char **argv)
{
  if (argc < 6) {
    std::cout << "________________________________________________________________________________________________" << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -      Computes the thresholds to be used for Hamming embedding.                             - " << std::endl;
    std::cout << " -                       written 2015-2016 by Torsten Sattler (sattlert@inf.ethz.ch)          - " << std::endl;
    std::cout << " -    copyright ETH Zurich, 2015-2016                                                         - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " - usage: compute_hamming_thresholds list_bins projection out sift_type nn                    - " << std::endl;
    std::cout << " - Parameters:                                                                                - " << std::endl;
    std::cout << " -  list_bins                                                                                 - " << std::endl;
    std::cout << " -     A text file containing the filenames of all descriptor files.                          - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  projection                                                                                - " << std::endl;
    std::cout << " -     The filename of a text file containing the projection matrix.                          - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  out                                                                                       - " << std::endl;
    std::cout << " -     Filename of the text file into which both the thresholds and the projection matrix are - " << std::endl;
    std::cout << " -     written. The resulting file can directly be used with build_partial_index.             - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  sift_type                                                                                 - " << std::endl;
    std::cout << " -     How the features were generated: 0 - VL_FEAT, 1 - VGG binary, 2 - HesAff binary        - " << std::endl;
    std::cout << " -     For types 0 and 2, the program assumes that the descriptor entries are stored as uint8 - " << std::endl;
    std::cout << " -     while floats were used for type 1.                                                     - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  nn                                                                                        - " << std::endl;
    std::cout << " -     The number of nearest neighboring words.                                               - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << "________________________________________________________________________________________________" << std::endl;
    return -1;
  }
  
  // Loads the Hamming embedding thresholds and the projection matrix.
  const int num_words = 200000;
  Eigen::Matrix<float, 64, Eigen::Dynamic> he_thresholds;
  he_thresholds.resize(64, num_words);
  Eigen::Matrix<float, 64, 128, Eigen::RowMajor> projection_matrix;

  int sift_type = atoi(argv[4]);
  const int num_nn_words = atoi(argv[5]);
  const bool invert_affine_matrix_when_loading_features = (sift_type == 0);

  {
    std::ifstream ifs(argv[2], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot read the projection "
                << "matrix from " << argv[2] << std::endl;
      return -1;
    }

    // Loads the projection matrix.
    for (int i = 0; i < 64; ++i) {
      for (int j = 0; j < 128; ++j) {
        ifs >> projection_matrix(i, j);
      }
    }

    ifs.close();
  }

  ////
  // Loads the list of keypoint names.
  std::vector<std::string> keypoint_filenames;
  {
    std::ifstream ifs(argv[1], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot read the list of keypoint files from "
                << argv[1] << std::endl;
    }

    while (!ifs.eof()) {
      std::string s;
      ifs >> s;
      if (keypoint_filenames.empty()) {
        keypoint_filenames.push_back(s);
      } else if (keypoint_filenames.back().compare(s) != 0) {
        keypoint_filenames.push_back(s);
      }
    }
    keypoint_filenames.pop_back();

    ifs.close();
  }
  int num_images = static_cast<int>(keypoint_filenames.size());
  std::cout << " Found " << num_images << " database images " << std::endl;


  ////
  // Loads for each word up to 2k words and then computes the thresholds.
  std::vector<std::vector<std::vector<float> > > entries_per_word(num_words);
  for (int i = 0; i < num_words; ++i) {
    entries_per_word[i].resize(64);
    for (int j = 0; j < 64; ++j) entries_per_word[i][j].clear();
  }
  std::vector<int> num_desc_per_word(num_words, 0);
  int num_missing_words = num_words;

  std::vector<int> randomly_permuted_db_ids(num_images);
  for (int i = 0; i < num_images; ++i) {
    randomly_permuted_db_ids[i] = i;
  }
  std::random_shuffle(randomly_permuted_db_ids.begin(),
                      randomly_permuted_db_ids.end());

  std::cout << " Determining relevant images per word " << std::endl;
  const int kNumDesiredDesc = 10000;
  for (int i = 0; i < num_images; ++i) {
    if (num_missing_words == 0) break;

    int id = randomly_permuted_db_ids[i];
    std::vector<geometric_burstiness::AffineSIFTKeypoint> keypoints;
    std::vector<std::vector<uint32_t> > words;
    Eigen::Matrix<float, 128, Eigen::Dynamic> descriptors_float;
    Eigen::Matrix<uint8_t, 128, Eigen::Dynamic> descriptors_uint8_t;

    bool loaded = false;
    if (sift_type == 0 || sift_type == 2) {
      loaded = geometric_burstiness::LoadAffineSIFTFeaturesAndMultipleNNWordAssignments<uint8_t>(
        keypoint_filenames[id], invert_affine_matrix_when_loading_features,
        num_nn_words, &keypoints, &words, &descriptors_uint8_t);
    } else {
      loaded = geometric_burstiness::LoadAffineSIFTFeaturesAndMultipleNNWordAssignments<float>(
        keypoint_filenames[id], invert_affine_matrix_when_loading_features,
        num_nn_words, &keypoints, &words, &descriptors_float);
    }
    if (!loaded) {
      std::cerr << " ERROR: Cannot load the descriptors for image " << id
                << " from " << keypoint_filenames[id] << std::endl;
      return -1;
    }

    int num_features = static_cast<int>(keypoints.size());

    if (num_features == 0) continue;

    for (int j = 0; j < num_features; ++j) {
      const int w = static_cast<int>(words[j][0]);
      if (num_desc_per_word[w] >= kNumDesiredDesc) continue;

      Eigen::Matrix<float, 128, 1> sift;
      if (sift_type == 0 || sift_type == 2) {
        sift = descriptors_uint8_t.col(j).cast<float>();
      } else {
        sift = descriptors_float.col(j);
      }
      Eigen::Matrix<float, 64, 1> proj_sift = projection_matrix * sift;

      for (int k = 0; k < 64; ++k) {
        entries_per_word[w][k].push_back(proj_sift[k]);
      }
      num_desc_per_word[w] += 1;

      if (num_desc_per_word[w] == kNumDesiredDesc) --num_missing_words;
    }

    if (i % 100 == 0) std::cout << "\r " << i << std::flush;
  }
  std::cout << std::endl;

  ////
  // For each word, computes the thresholds.
  std::cout << " Computing the thresholds per word " << std::endl;
  for (int i = 0; i < num_words; ++i) {
    int num_desc = num_desc_per_word[i];

    if (num_desc == 0) {
      std::cout << " WARNING: FOUND EMPTY WORD " << i << std::endl;
      he_thresholds.col(i) = Eigen::Matrix<float, 64, 1>::Zero();
    } else {
      const int median_element = num_desc / 2;
      for (int k = 0; k < 64; ++k) {
        std::nth_element(entries_per_word[i][k].begin(),
                         entries_per_word[i][k].begin() + median_element,
                         entries_per_word[i][k].end());
        he_thresholds(k, i) = entries_per_word[i][k][median_element];
      }
    }

    if (i % 1000 == 0) std::cout << "\r word " << i << std::flush;
  }
  std::cout << " done" << std::endl;

  ////
  // Writes out the data.
  std::ofstream ofs(argv[3], std::ios::out);
  if (!ofs.is_open()) {
    std::cerr << "ERROR: Cannot write to " << argv[3] << std::endl;
    return -1;
  }

  ofs << num_words << " 128 64" << std::endl;
  for (int i = 0; i < num_words; ++i) {
    for (int j = 0; j < 64; ++j) {
      ofs << he_thresholds(j, i) << " ";
    }
    ofs << std::endl;
  }
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 128; ++j) {
      ofs << projection_matrix(i, j) << " ";
    }
    ofs << std::endl;
  }
  ofs.close();

  return 0;
}

