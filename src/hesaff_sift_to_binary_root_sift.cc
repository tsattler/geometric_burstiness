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
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace geometric_burstiness_hesaff {
struct HesAffRootSIFT {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Specifies the parameters of an ellipse such that a point (u,v) on the
  // ellipse satisfies a(u-x)^2 + 2b(u-x)(v-y) + c(v-y)^2 = 1.
  float x;
  float y;
  float a;
  float b;
  float c;

  Eigen::Matrix<uint8_t, 128, 1> descriptor;
};

typedef std::vector<HesAffRootSIFT,
                    Eigen::aligned_allocator<HesAffRootSIFT> > HesAffRootSIFTVec;

// Loads a .sift file generated with the binary from Michal Perdoch (see
// https://github.com/perdoch/hesaff) from a text file. Converts the SIFT
// descriptors to RootSIFT descriptors while loading.
bool LoadHesAffFile(const std::string& filename,
                    HesAffRootSIFTVec* sifts) {
  sifts->clear();

  std::ifstream ifs(filename.c_str(), std::ios::in);
  if (!ifs.is_open()) {
    std::cerr << " ERROR: Cannot read features from " << filename << std::endl;
    return false;
  }

  int desc_dim = 0;
  ifs >> desc_dim;
  if (desc_dim != 128) {
    std::cerr << " ERROR: Expecting 128 descriptor entries instead of "
              << desc_dim << std::endl;
    return false;
  }

  int num_features = 0;
  ifs >> num_features;
  if (num_features < 0) {
    std::cerr << " ERROR: Expecting positive number of features " << std::endl;
    std::cout << filename << std::endl;
    ifs.close();
    return false;
  }
  if (num_features == 0) {
    ifs.close();
    return true;
  }

  sifts->resize(num_features);
  HesAffRootSIFTVec& s_vec = *sifts;

  for (int i = 0; i < num_features; ++i) {
    ifs >> s_vec[i].x >> s_vec[i].y >> s_vec[i].a >> s_vec[i].b >> s_vec[i].c;

    Eigen::Matrix<float, 128, 1> desc;
    for (int j = 0; j < 128; ++j) {
      int d = 0;
      ifs >> d;
      desc[j] = static_cast<float>(d);
    }

    // Conversion to RootSIFT.
    desc.normalize();
    float L1norm = 0.0f;
    for (int j = 0; j < 128; ++j) L1norm += desc[j];

    desc /= L1norm;
    for (int j = 0; j < 128; ++j) desc[j] = std::sqrt(desc[j]);

    // Stores the descriptor.
    for (int j = 0; j < 128; ++j) {
      float d = 512.0f * desc[j] + 0.5f;
      if (d > 255.5f) {
        std::cout << " Large descriptor element: " << d << std::endl;
        d = 255.0f;
      }
      s_vec[i].descriptor[j] = static_cast<uint8_t>(d);
    }
  }

  ifs.close();

  return true;
}

// Writes a set of RootSIFT features to a binary file. The file format is
// similar to the one of HesAff:
// number_of_features (uint32_t)
// for each feature : x y a b c (all float) 128 descriptor entries (uint8_t)
bool SaveBinaryHesAffFile(const std::string& filename,
                          const HesAffRootSIFTVec& sifts) {
  std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    std::cerr << " ERROR: Cannot write features to " << filename << std::endl;
    return false;
  }

  uint32_t num_features = static_cast<uint32_t>(sifts.size());
  ofs.write((const char*) &num_features, sizeof(uint32_t));

  for (uint32_t i = 0; i < num_features; ++i) {
    float x = sifts[i].x;
    float y = sifts[i].y;
    float a = sifts[i].a;
    float b = sifts[i].b;
    float c = sifts[i].c;

    ofs.write((const char*) &x, sizeof(float));
    ofs.write((const char*) &y, sizeof(float));
    ofs.write((const char*) &a, sizeof(float));
    ofs.write((const char*) &b, sizeof(float));
    ofs.write((const char*) &c, sizeof(float));

    for (int j = 0; j < 128; ++j) {
      uint8_t d = static_cast<uint8_t>(sifts[i].descriptor[j]);
      ofs.write((const char*) &d, sizeof(uint8_t));
    }
  }
  ofs.close();

  return true;
}

}  // namespace geometric_burstiness_hesaff

int main(int argc, char **argv) {

  if (argc < 3) {
    std::cout << "________________________________________________________________________________________________" << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -    Loads SIFTs from a text file, converts them to RootSIFT and saves them to a binary file - " << std::endl;
    std::cout << " -                       written 2016 by Torsten Sattler (sattlert@inf.ethz.ch)               - " << std::endl;
    std::cout << " -    copyright ETH Zurich, 2016                                                              - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " - usage: hesaff_sift_to_binary_root_sift list_in list_out                                    - " << std::endl;
    std::cout << " - Parameters:                                                                                - " << std::endl;
    std::cout << " -  list_sifts                                                                                - " << std::endl;
    std::cout << " -     A text file containing the filenames of all descriptor files.                          - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << " -  list_out                                                                                  - " << std::endl;
    std::cout << " -     A text file containing the list of filenames of the binary files to be created.        - " << std::endl;
    std::cout << " -                                                                                            - " << std::endl;
    std::cout << "________________________________________________________________________________________________" << std::endl;
    return -1;
  }

  std::vector<std::string> input_names;
  {
    std::ifstream ifs(argv[1], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot read from " << argv[1] << std::endl;
      return -1;
    }

    while (!ifs.eof()) {
      std::string s = "";
      ifs >> s;
      if (!s.empty()) {
        if (!input_names.empty()) {
          if (input_names.back().compare(s) == 0) continue;
        }
        input_names.push_back(s);
      }
    }
    ifs.close();
  }
  int num_files = static_cast<int>(input_names.size());

  std::vector<std::string> output_names;
  {
    std::ifstream ifs(argv[2], std::ios::in);
    if (!ifs.is_open()) {
      std::cerr << "ERROR: Cannot read from " << argv[2] << std::endl;
      return -1;
    }

    for (int i = 0; i < num_files; ++i) {
      std::string s = "";
      ifs >> s;
      output_names.push_back(s);
    }
    ifs.close();
  }
  std::cout << " Processing " << num_files << " files " << std::endl;

  for (int i = 0; i < num_files; ++i) {
    geometric_burstiness_hesaff::HesAffRootSIFTVec sifts;
    if (geometric_burstiness_hesaff::LoadHesAffFile(input_names[i], &sifts)) {
      geometric_burstiness_hesaff::SaveBinaryHesAffFile(output_names[i], sifts);
    }

    if (i % 100 == 0) std::cout << "  \r " << i << std::flush;
  }
  std::cout << " ... done" << std::endl;


  return 0;
}

