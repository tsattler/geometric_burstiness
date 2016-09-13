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

#ifndef GEOMETRIC_BURSTINESS_SRC_HESAFF_SIFT_H_
#define GEOMETRIC_BURSTINESS_SRC_HESAFF_SIFT_H_

#include <fstream>
#include <iostream>
#include <stdint.h>
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
    std::cout << " This file is causing problems: " << filename << std::endl;
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

// Similar function as above, only that the function now also writes out visual
// word assignments.
bool SaveBinaryHesAffWithWordsFile(
    const std::string& filename, const int num_words,
    const HesAffRootSIFTVec& sifts,
    const std::vector<std::vector<uint32_t> >& word_assignemnts) {
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

    for (int j = 0; j < num_words; ++j) {
      uint32_t w = word_assignemnts[i][j];
      ofs.write((const char*) &w, sizeof(uint32_t));
    }

    for (int j = 0; j < 128; ++j) {
      uint8_t d = static_cast<uint8_t>(sifts[i].descriptor[j]);
      ofs.write((const char*) &d, sizeof(uint8_t));
    }
  }
  ofs.close();

  return true;
}

// Loads a set of Hessian Affine RootSIFT features from a binary file.
bool LoadBinaryHesAffFile(const std::string& filename,
                          HesAffRootSIFTVec* sifts) {
  sifts->clear();
  std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << " ERROR: Cannot read features from " << filename << std::endl;
    return false;
  }

  uint32_t num_features = 0;
  ifs.read((char*) &num_features, sizeof(uint32_t));
  if (num_features == 0) {
    ifs.close();
    return true;
  }

  sifts->resize(num_features);
  for (uint32_t i = 0; i < num_features; ++i) {
    float x = 0.0;
    float y = 0.0;
    float a = 0.0;
    float b = 0.0;
    float c = 0.0;

    ifs.read((char*) &x, sizeof(float));
    ifs.read((char*) &y, sizeof(float));
    ifs.read((char*) &a, sizeof(float));
    ifs.read((char*) &b, sizeof(float));
    ifs.read((char*) &c, sizeof(float));

    (*sifts)[i].x = x;
    (*sifts)[i].y = y;
    (*sifts)[i].a = a;
    (*sifts)[i].b = b;
    (*sifts)[i].c = c;

    for (int j = 0; j < 128; ++j) {
      uint8_t d = 0;;
      ifs.read((char*) &d, sizeof(uint8_t));
      (*sifts)[i].descriptor[j] = static_cast<uint8_t>(d);
    }
  }
  ifs.close();

  return true;
}

}  // namespace geometric_burstiness_hesaff


#endif  // GEOMETRIC_BURSTINESS_SRC_HESAFF_SIFT_H_
