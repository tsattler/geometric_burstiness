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
#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <stdint.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_set>
#include <vector>

#include "hesaff_sift.h"

#include <flann/flann.hpp>

int main(int argc, char **argv) {
  // Resets the random number generator for deterministic behavior.
  // This is required to ensure that FLANN always generates the same search
  // tree every time the programm is called.
  srand(1);

  if (argc < 6) {
    std::cout << "________________________________________________________________________________________________" << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -    Assigns features extracted with a hessian affine detector to visual words.              - " << std::endl;
    std::cout << " -                       written 2015 by Torsten Sattler (sattlert@inf.ethz.ch)               - " << std::endl;
    std::cout << " -    copyright ETH Zurich, 2015-2016                                                         - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " - usage: compute_word_assignments vocabulary num_words num_nn list postfix                   - " << std::endl;
    std::cout << " - Parameters:                                                                                - " << std::endl;
    std::cout << " -  vocabulary                                                                                - " << std::endl;
    std::cout << " -     A text file containing the visual vocabulary, one word per row.                        - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  num_words                                                                                 - " << std::endl;
    std::cout << " -     The number of words in the vocabulary.                                                 - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  num_nn                                                                                    - " << std::endl;
    std::cout << " -     The number of closest words to find.                                                   - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  list                                                                                      - " << std::endl;
    std::cout << " -     A text file containing the list of binary files (created with hesaff_sift_to_binary_   - " << std::endl;
    std::cout << " -     root_sift) that should be used for the assigments.                                     - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  postfix                                                                                   - " << std::endl;
    std::cout << " -     For each file a in list, a file a.postfix is written out.                              - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << "________________________________________________________________________________________________" << std::endl;
    return -1;
  }

  uint64_t num_words = static_cast<uint64_t>(atoi(argv[2]));
  if (num_words <= 0) {
    std::cerr << " The number of words needs to be positive " << std::endl;
    return -1;
  }

  // Loads the visual words directly into a FLANN matrix. The entries are
  // rounded to uint8_t values to safe memory.
  std::cout << " Loading the visual words" << std::endl;
  flann::Matrix<uint8_t> words(
      new uint8_t[num_words * static_cast<uint64_t>(128)], num_words, 128);

  {
    std::ifstream ifs(argv[1], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot read from " << argv[1] << std::endl;
      return -1;
    }

    for (uint64_t i = 0; i < num_words; ++i) {
      for (int j = 0; j < 128; ++j) {
        float d = 0.0f;
        ifs >> d;
        words[i][j] = static_cast<uint8_t>(std::min(d + 0.5f, 255.0f));
      }
    }
    ifs.close();
  }
  std::cout << "  ... done " << std::endl;

  // Builds the search tree.
  std::cout << " Building the index " << std::endl;
  flann::KDTreeIndexParams tree_params(128);
  tree_params["random_seed"] = static_cast<long>(0);
  std::cout << "  Random seed: " << tree_params["random_seed"] << std::endl;
  flann::Index<flann::L2<uint8_t> > index(words, tree_params);
  index.buildIndex();
  std::cout << "  ... done " << std::endl;

  int num_nn = atoi(argv[3]);
  if (num_nn <= 0) {
    std::cerr << " ERROR: The number of desired nearest words needs to be "
              << "positive " << std::endl;
    return -1;
  }

  ////
  // Loads the images used for querying the tree.
  std::vector<std::string> query_names;
  {
    std::ifstream ifs(argv[4], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "WARNING: Cannot read from " << argv[4] << std::endl;
    } else {
      while (!ifs.eof()) {
        std::string s = "";
        ifs >> s;
        if (!s.empty()) {
          if (!query_names.empty()) {
            if (query_names.back().compare(s) == 0) continue;
          }
          query_names.push_back(s);
        }
      }
    }
    ifs.close();
  }
  int num_query = static_cast<int>(query_names.size());
  std::cout << " Processing " << query_names.size() << " images" << std::endl;


  for (int i = 0; i < num_query; ++i) {
    geometric_burstiness_hesaff::HesAffRootSIFTVec sifts;
    if (geometric_burstiness_hesaff::LoadBinaryHesAffFile(query_names[i], &sifts)) {
      int num_sifts = static_cast<int>(sifts.size());
      std::vector<std::vector<uint32_t> > words;

      if (num_sifts > 0) {
        words.resize(num_sifts);
        flann::Matrix<uint8_t> query(new uint8_t[num_sifts * 128],
                                     num_sifts, 128);
        for (int j = 0; j < num_sifts; ++j) {
          for (int k = 0; k < 128; ++k) {
            query[j][k] = static_cast<uint8_t>(sifts[j].descriptor[k]);
          }
        }

        flann::Matrix<int> indices(new int[num_sifts * num_nn], num_sifts,
                                   num_nn);
        flann::Matrix<float> distances(new float[num_sifts * num_nn], num_sifts,
                                       num_nn);
        index.knnSearch(query, indices, distances, num_nn,
                        flann::SearchParams(1024));

        for (int j = 0; j < num_sifts; ++j) {
          words[j].clear();
          for (int k = 0; k < num_nn; ++k) {
            words[j].push_back(static_cast<uint32_t>(indices[j][k]));
            if ((indices[j][k] < 0
                || static_cast<uint64_t>(indices[j][k]) >= num_words)) {
              std::cerr << " ERROR: Found invalid word " << indices[j][k]
                        << std::endl;
            }
          }
        }

        delete [] query.ptr();
        delete [] indices.ptr();
        delete [] distances.ptr();
      }

      std::string outname(query_names[i]);
      outname.append(".");
      outname.append(argv[5]);

      if (!geometric_burstiness_hesaff::SaveBinaryHesAffWithWordsFile(
        outname, num_nn, sifts, words)) {
        std::cerr << " ERROR: Cannot write to " << outname << std::endl;
      }
    } else {
      std::cerr << " ERROR: Could not load " << query_names[i] << std::endl;
    }

    if (i % 1 == 0) std::cout << "  \r " << i << std::flush;
  }

  delete [] words.ptr();

  return 0;
}

