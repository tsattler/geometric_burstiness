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

int main (int argc, char **argv)
{
  if (argc < 2) {
    std::cout << "________________________________________________________________________________________________" << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -      Given an inverted index, computes visual word weights and saves them to disk.         - " << std::endl;
    std::cout << " -                       written 2015 by Torsten Sattler (sattlert@inf.ethz.ch)               - " << std::endl;
    std::cout << " -    copyright ETH Zurich, 2015-2016                                                         - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " - usage: compute_index_weights index                                                         - " << std::endl;
    std::cout << " - Parameters:                                                                                - " << std::endl;
    std::cout << " -  index                                                                                     - " << std::endl;
    std::cout << " -     The filename of an inverted index created with build_index. This program creates a     - " << std::endl;
    std::cout << " -     file index.weights storing all idf-weights and normalization constants.                - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << "________________________________________________________________________________________________" << std::endl;
    return -1;
  }

  geometric_burstiness::InvertedIndex<64> inverted_index;
  std::string index_file(argv[1]);
  if (!inverted_index.LoadInvertedIndex(index_file)) {
    std::cerr << "ERROR: Cannot load the inverted index from " << index_file
              << std::endl;
    return -1;
  }

  if (!inverted_index.ComputeWeightsAndNormalizationConstants()) {
    std::cerr << "ERROR: Failed to compute the weights and constants"
              << std::endl;
    return -1;
  }
  
  std::string outfile(index_file);
  outfile.append(".weights");
  if (!inverted_index.SaveWeightsAndConstants(outfile)) {
    std::cerr << "ERROR: Could not save the weights to " << outfile
              << std::endl;
    return -1;
  }

  return 0;
}

