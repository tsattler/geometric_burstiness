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
    std::cout << " -      Builds an inverted index for query.                                                   - " << std::endl;
    std::cout << " -                       written 2015-2016 by Torsten Sattler (sattlert@inf.ethz.ch)          - " << std::endl;
    std::cout << " -    copyright ETH Zurich, 2015-2016                                                         - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " - usage: build_partial_index list_bin hamming_and_projection out sift_type nn                - " << std::endl;
    std::cout << " - Parameters:                                                                                - " << std::endl;
    std::cout << " -  list_bin                                                                                  - " << std::endl;
    std::cout << " -     A text file containing the filenames of all descriptor files.                          - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  hamming_and_projection                                                                    - " << std::endl;
    std::cout << " -     The filename of a text file containing the Hamming embedding thresholds and the        - " << std::endl;
    std::cout << " -     projection matrix. The file has the format:                                            - " << std::endl;
    std::cout << " -       num_words num_dimensions num_bits                                                    - " << std::endl;
    std::cout << " -       for each word, one hamming threshold, each stored in its own line                    - " << std::endl;
    std::cout << " -       A 64x128 projection matrix stored in row-major order.                                - " << std::endl;
    std::cout << " -     Here, num_words is the number of visual words, num_dimensions the dimensionality of    - " << std::endl;
    std::cout << " -     the descriptors (128 for SIFT), and num_bits is the number of bits to use for Hamming  - " << std::endl;
    std::cout << " -     embedding (has to be 64).                                                              - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  out                                                                                       - " << std::endl;
    std::cout << " -     Filename of the binary file into which to write the inverted index.                    - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  sift_type                                                                                 - " << std::endl;
    std::cout << " -     How the features were generated: 0 - VL_FEAT, 1 - VGG binary, 2 - HesAff binary.       - " << std::endl;
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
  Eigen::Matrix<float, 64, Eigen::Dynamic> he_thresholds;
  Eigen::Matrix<float, 64, 128, Eigen::RowMajor> projection_matrix;
  int num_words = 0;
  {
    std::ifstream ifs(argv[2], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot read the Hamming thresholds and projection "
                << "matrix from " << argv[2] << std::endl;
      return -1;
    }

    int num_dimensions = 0;
    int num_bits = 0;
    ifs >> num_words >> num_dimensions >> num_bits;
    std::cout << " num words: " << num_words << std::endl;
    std::cout << " num dims: " << num_dimensions << std::endl;
    if (num_dimensions != 128) {
      std::cerr << "ERROR: The number of dimensions should be 128" << std::endl;
      return -1;
    }
    std::cout << " num bits: " << num_bits << std::endl;
    if (num_bits != 64) {
      std::cerr << "ERROR: The number of bits should be 64" << std::endl;
      return -1;
    }
    // Loads the thresholds for Hamming embedding.
    he_thresholds.resize(64, num_words);
    for (int i = 0; i < num_words; ++i) {
      for (int j = 0; j < 64; ++j) {
        ifs >> he_thresholds(j, i);
      }
    }

    // Loads the projection matrix.
    for (int i = 0; i < 64; ++i) {
      for (int j = 0; j < 128; ++j) {
        ifs >> projection_matrix(i, j);
      }
    }

    ifs.close();
  }

  // Parameter settings.
  geometric_burstiness::InvertedIndex<64> inverted_index;
  inverted_index.InitializeIndex(num_words);
  std::cout << " Index initialized with " << num_words << " words" << std::endl;
  
  inverted_index.SetProjectionMatrix(projection_matrix);
  std::cout << " Projection matrix initialized" << std::endl;

  // Assigns the thresholds to the inverted files.
  if (!inverted_index.SetHammingThresholds(he_thresholds)) {
    std::cerr << "ERROR: Could not assign the thresholds " << std::endl;
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

  // Computes a mean vector.
  const int sift_type = atoi(argv[4]);
  bool invert_affine_matrix_when_loading_features = (sift_type == 0);
  const int num_nn_words = atoi(argv[5]);

  ////
  // Loads all images, creates and assigns the inverted file entries, sorts all
  // inverted files, and finally estimates the descriptor space densities.
  std::cout << " Creating inverted file entries " << std::endl;
  int total_number_entries = 0;
  if (num_nn_words <= 0) {
    std::cerr << "ERROR: The number of nearest neighboring words needs to be "
              << "positive" << std::endl;
    return -1;
  }
  for (int i = 0; i < num_images; ++i) {
    std::vector<geometric_burstiness::AffineSIFTKeypoint> keypoints;
    std::vector<std::vector<uint32_t> > words;
    Eigen::Matrix<float, 128, Eigen::Dynamic> descriptors_float;
    Eigen::Matrix<uint8_t, 128, Eigen::Dynamic> descriptors_uint8_t;

    bool loaded = false;
    if (sift_type == 0 || sift_type == 2) {
      loaded = geometric_burstiness::LoadAffineSIFTFeaturesAndMultipleNNWordAssignments<uint8_t>(
        keypoint_filenames[i], invert_affine_matrix_when_loading_features,
        num_nn_words, &keypoints, &words, &descriptors_uint8_t);
    } else {
      loaded = geometric_burstiness::LoadAffineSIFTFeaturesAndMultipleNNWordAssignments<float>(
        keypoint_filenames[i], invert_affine_matrix_when_loading_features,
        num_nn_words, &keypoints, &words, &descriptors_float);
    }
    if (!loaded) {
      std::cerr << " ERROR: Cannot load the descriptors for image " << i
                << " from " << keypoint_filenames[i] << std::endl;
      return -1;
    }

    int num_features = static_cast<int>(keypoints.size());

    if (num_features == 0) continue;

    for (int j = 0; j < num_features; ++j, ++total_number_entries) {
      geometric_burstiness::InvertedFileEntry<64> entry;
      entry.image_id = i;
      entry.x = keypoints[j].x[0];
      entry.y = keypoints[j].x[1];
      entry.a = keypoints[j].a;
      entry.b = keypoints[j].b;
      entry.c = keypoints[j].c;

      const int w = static_cast<int>(words[j][0]);
      if (w < 0 || w >= num_words) {
        std::cerr << "ERROR: Impossible word " << w << " ( " << words[j][0]
                  << " ) " << std::endl;
        return -1;
      }
      Eigen::Matrix<float, 128, 1> sift;
      if (sift_type == 0 || sift_type == 2) {
        sift = descriptors_uint8_t.col(j).cast<float>();
      } else {
        sift = descriptors_float.col(j);
      }
      inverted_index.AddEntry(entry, w, sift);
    }
    if (i % 100 == 0) std::cout << "\r " << i << std::flush;
  }
  std::cout << std::endl;
  std::cout << " The index contains " << total_number_entries << " entries "
            << "in total" << std::endl;

  std::cout << " Estimates the descriptor space densities " << std::endl;
  inverted_index.EstimateDescriptorSpaceDensities();

  inverted_index.FinalizeIndex();
  std::cout << " Inverted index finalized" << std::endl;
  
  std::string outfile(argv[3]);
  if (!inverted_index.SaveInvertedIndex(outfile)) {
    std::cerr << "ERROR: Could not write the inverted index to "
              << outfile << std::endl;
    return -1;
  }

  return 0;
}

