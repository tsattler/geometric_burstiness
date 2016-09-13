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

#ifndef GEOMETRIC_BURSTINESS_SRC_STRUCTURES_H_
#define GEOMETRIC_BURSTINESS_SRC_STRUCTURES_H_

#include <bitset>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

// Defines multiple common structures.
namespace geometric_burstiness {
// Different comparison operators for std::pairs.
template <typename T1, typename T2>
inline bool CmpPairSecondLess(const std::pair<T1, T2>& a,
                              const std::pair<T1, T2>& b) {
  return a.second < b.second;
}

template<typename T1, typename T2>
inline bool CmpPairSecondGreater(const std::pair<T1, T2>& a,
                                 const std::pair<T1, T2>& b) {
  return a.second > b.second;
}

// Models an inverted file entry. The template defines the dimensionality of
// the binary string used to approximate the descriptor.
template <int N>
struct InvertedFileEntry {
  int image_id;
  float x;
  float y;
  // Defines an elliptical regions such that ax^2 + bxy + cy^2 = 1 holds for all
  // points on the ellipse.
  float a;
  float b;
  float c;
  std::bitset<N> descriptor;
};

template <int N>
float GetAreaOfEllipse(const InvertedFileEntry<N>& e) {
  return 2.0f * M_PI / sqrtf(4.0 * e.a * e.c + e.b * e.b);
}

// Reads an inverted file entry from a binary file.
// Notice that bitstrings of length >64 are currently not supported.
template <int N>
bool ReadInvertedFileEntryFromBinary(std::ifstream* ifs,
                                     InvertedFileEntry<N>* entry) {
  if (N > 64) {
    std::cout << "ERROR: Dimensionality " << N << " too large " << std::endl;
    return false;
  }
  if (sizeof(unsigned long) < 8) {
    std::cerr << " ERROR: Expected unsigned long to be at least 8 byte "
              << std::endl;
    return false;
  }

  int32_t image_id = 0;
  ifs->read((char*) &image_id, sizeof(int32_t));
  entry->image_id = static_cast<int>(image_id);

  float f = 0.0f;
  ifs->read((char*) &f, sizeof(float));
  entry->x = f;
  ifs->read((char*) &f, sizeof(float));
  entry->y = f;
  ifs->read((char*) &f, sizeof(float));
  entry->a = f;
  ifs->read((char*) &f, sizeof(float));
  entry->b = f;
  ifs->read((char*) &f, sizeof(float));
  entry->c = f;

  uint64_t bin_desc = 0;
  ifs->read((char*) &bin_desc, sizeof(uint64_t));
  entry->descriptor = std::bitset<N>(bin_desc);

  return true;
}

// Writes an inverted file entry into a binary file.
// Notice that bitstrings of length >64 are currently not supported.
template <int N>
bool WriteInvertedFileEntryFromBinary(const InvertedFileEntry<N>& entry,
                                      std::ofstream* ofs) {
  if (N > 64) return false;
  if (sizeof(unsigned long) < 8) return false;

  int32_t image_id = static_cast<int32_t>(entry.image_id);
  ofs->write((const char*) &image_id, sizeof(int32_t));

  float f = entry.x;
  ofs->write((const char*) &f, sizeof(float));

  f = entry.y;
  ofs->write((const char*) &f, sizeof(float));

  f = entry.a;
  ofs->write((const char*) &f, sizeof(float));

  f = entry.b;
  ofs->write((const char*) &f, sizeof(float));

  f = entry.c;
  ofs->write((const char*) &f, sizeof(float));

  uint64_t bin_desc = static_cast<uint64_t>(entry.descriptor.to_ulong());
  ofs->write((const char*) &bin_desc, sizeof(uint64_t));

  return true;
}

// Represents a query descriptor.
template <int N>
struct QueryDescriptor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int feature_id;
  float x;
  float y;
  // Defines an elliptical regions such that ax^2 + bxy + cy^2 = 1 holds for all
  // points on the ellipse.
  float a;
  float b;
  float c;
  // The descriptor projected into an N-dimensional space.
  Eigen::Matrix<float, N, 1> proj_desc;
  // The nearest neighboring words, sorted in ascending order of distance.
  std::vector<int> relevant_word_ids;
  std::vector<size_t> max_hamming_distance_per_word;
};

// Defines a 2D-2D match between a query feature and a database feature.
// To save memory, we only store indices to the query and database feature.
// For the later, we store the word id and the feature's index in the inverted
// file.
struct Match2D2D {
  Match2D2D(int q_id, int db_w_id, int db_f_index, float w)
      : query_feature_id(q_id),
        db_feature_word(db_w_id),
        db_feature_index(db_f_index),
        weight(w) {}

  int query_feature_id;
  int db_feature_word;
  int db_feature_index;
  float weight;
};

// Function to sort InvertedFileEntries based on their image id.
template <int N>
inline bool CmpInvertedFileEntries(const InvertedFileEntry<N>& entry_1,
                                   const InvertedFileEntry<N>& entry_2) {
  return entry_1.image_id < entry_2.image_id;
}

// Voting score and matches together with the image id.
struct ImageScore {
  int image_id;
  float voting_weight;
  std::vector<Match2D2D> matches;
};

// Function to compare two ImageScores, returning true if the score of the
// first image is larger than the score of the second.
inline bool CmpImageScores(const ImageScore& s1, const ImageScore& s2) {
  return s1.voting_weight > s2.voting_weight;
}

// Implements the weighting function used to weight a vote. See Eqn. 4 in
// Arandjelovic, Zisserman. DisLocation: Scalable descriptor distinctiveness for
// location recognition. ACCV 2014.
// Again, the template is the length of the Hamming embedding vectors.
template <int N>
class VoteWeightingFunction {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VoteWeightingFunction() {
    // Fills the look-up table.
    for (int i = 0; i <= N; ++i) {
      double sigma = static_cast<double>(i);
      double sigma_squared = sigma * sigma;
      double thresh = 1.5 * sigma;
      // Computes the function for all possible Hamming distance values.
      for (int j = 0; j <= N; ++j) {
        double h = static_cast<double>(j);
        if (h > thresh) {
          function_look_up_table_(j, i) = 0.0f;
        } else {
          double expnnt = h * h / sigma_squared;
          function_look_up_table_(j, i) = static_cast<float>(std::exp(-expnnt));
        }
      }
    }
  }

  // Returns the weight for Hamming distance h and standard deviation sigma.
  // Does not perform a range check when performing the look-up.
  float GetWeight(size_t h, int sigma) const {
    return function_look_up_table_(h, sigma);
  }

 private:
  // In order to avoid wasting computations, we once compute a look-up table
  // storing all function values for all possible values of the standard
  // deviation \sigma. This is implemented as a (N + 1) x (N + 1) matrix.
  Eigen::Matrix<float, N + 1, N + 1> function_look_up_table_;
};

// Represents a Hamming space of dimensionality N. Stores for each possible
// entry a list of all other Hamming vectors in ascending order of Hamming
// distance.
template <int N>
class HammingSpace {
 public:
  HammingSpace() {
    // Fills the table containing the entries.
    const int kNumPossibleVectors = 1 << N;
    std::cout << " Hamming space of dimension " << N << " contains "
              << kNumPossibleVectors << " possible values " << std::endl;
    distances_.resize(kNumPossibleVectors, kNumPossibleVectors);
    lists_.resize(kNumPossibleVectors, kNumPossibleVectors);
    for (int i = 0; i < kNumPossibleVectors; ++i) {
      std::bitset<N> b(static_cast<size_t>(i));
      std::vector<std::pair<int, int> > distances_and_ids(kNumPossibleVectors);
      for (int j = 0; j < kNumPossibleVectors; ++j) {
        std::bitset<N> bj(static_cast<size_t>(j));
        distances_(j, i) = static_cast<int>((b ^ bj).count());
        distances_and_ids[j].first = distances_(j, i);
        distances_and_ids[j].second = j;
      }
      std::sort(distances_and_ids.begin(), distances_and_ids.end(),
                std::less<std::pair<int, int> >());
      for (int j = 0; j < kNumPossibleVectors; ++j) {
        lists_(j, i) = distances_and_ids[j].second;
      }
    }
  }

  // Stores in the i-th column the distances of all possible binary strings to
  // the binary string corresponding to the number i.
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> distances_;
  // Stores in the i-th column the list of vector indices sorted in ascending
  // order of distance to the bitstring corresponding to the number i.
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> lists_;
};

// Represents a SIFT keypoint defined by its position, scale, and orientation.
struct SIFTKeypoint {
  Eigen::Vector2f x;
  float scale;
  float orientation;
};

// Represents a SIFT keypoint by its position and the affine parameters of an
// ellipse, such that a point (x, y)^T is on the ellipse if
// a * x^2 + b * xy + c * y^2 = 1.
struct AffineSIFTKeypoint {
  Eigen::Vector2f x;
  float a;
  float b;
  float c;
};

// Loads AffineSIFTKeypoints, their descriptors and visual word assignments
// from a binary file.
template <typename DescriptorType>
bool LoadAffineSIFTFeaturesAndWordAssignments(
    const std::string& filename, bool invert_matrix,
    std::vector<AffineSIFTKeypoint>* keypoints,
    std::vector<uint32_t>* word_assignments,
    Eigen::Matrix<DescriptorType, 128, Eigen::Dynamic>* descriptors) {
  keypoints->clear();
  word_assignments->clear();

  std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "ERROR: Cannot read the features from " << filename
              << std::endl;
  }

  uint32_t num_features;
  ifs.read((char*) &num_features, sizeof(uint32_t));

  if (num_features == 0) return true;

  keypoints->resize(num_features);
  word_assignments->resize(num_features);
  descriptors->resize(128, num_features);

  for (uint32_t i = 0; i < num_features; ++i) {
    float x = 0.0f;
    ifs.read((char*) &x, sizeof(float));
    float y = 0.0f;
    ifs.read((char*) &y, sizeof(float));
    float a = 0.0f;
    ifs.read((char*) &a, sizeof(float));
    float b = 0.0f;
    ifs.read((char*) &b, sizeof(float));
    float c = 0.0f;
    ifs.read((char*) &c, sizeof(float));

    (*keypoints)[i].x << x, y;

    if (invert_matrix) {
      Eigen::Matrix2f M;
      M << a, b, b, c;
      Eigen::Matrix2f M_inv = M.inverse();
      (*keypoints)[i].a = M_inv(0, 0);
      (*keypoints)[i].b = M_inv(0, 1) + M_inv(1, 0);
      (*keypoints)[i].c = M_inv(1, 1);
    } else {
      (*keypoints)[i].a = a;
      (*keypoints)[i].b = 2.0f * b;
      (*keypoints)[i].c = c;
    }

    uint32_t w = 0;
    ifs.read((char*) &w, sizeof(uint32_t));
    (*word_assignments)[i] = w;

    for (int j = 0; j < 128; ++j) {
      DescriptorType d;
      ifs.read((char*) &d, sizeof(DescriptorType));
      (*descriptors)(j, i) = d;
    }
  }

  ifs.close();

  return true;
}
  
// Loads AffineSIFTKeypoints, their descriptors and visual word assignments
// from a binary file.
template <typename DescriptorType>
bool LoadAffineSIFTFeaturesAndMultipleNNWordAssignments(
     const std::string& filename, bool invert_matrix, int num_nn,
     std::vector<AffineSIFTKeypoint>* keypoints,
     std::vector<std::vector<uint32_t> >* word_assignments,
     Eigen::Matrix<DescriptorType, 128, Eigen::Dynamic>* descriptors) {
  keypoints->clear();
  word_assignments->clear();
    
  std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "ERROR: Cannot read the features from " << filename
              << std::endl;
    return false;
  }
    
  uint32_t num_features;
  ifs.read((char*) &num_features, sizeof(uint32_t));
    
  if (num_features == 0) return true;
    
  keypoints->resize(num_features);
  word_assignments->resize(num_features);
  descriptors->resize(128, num_features);
    
  for (uint32_t i = 0; i < num_features; ++i) {
    float x = 0.0f;
    ifs.read((char*) &x, sizeof(float));
    float y = 0.0f;
    ifs.read((char*) &y, sizeof(float));
    float a = 0.0f;
    ifs.read((char*) &a, sizeof(float));
    float b = 0.0f;
    ifs.read((char*) &b, sizeof(float));
    float c = 0.0f;
    ifs.read((char*) &c, sizeof(float));
      
    (*keypoints)[i].x << x, y;

    if (invert_matrix) {
      Eigen::Matrix2f M;
      M << a, b, b, c;
      Eigen::Matrix2f M_inv = M.inverse();
      (*keypoints)[i].a = M_inv(0, 0);
      (*keypoints)[i].b = M_inv(0, 1) + M_inv(1, 0);
      (*keypoints)[i].c = M_inv(1, 1);
    } else {
      (*keypoints)[i].a = a;
      (*keypoints)[i].b = 2.0f * b;
      (*keypoints)[i].c = c;
    }
      
    (*word_assignments)[i].resize(num_nn);
    for (int j = 0; j < num_nn; ++j) {
      uint32_t w = 0;
      ifs.read((char*) &w, sizeof(uint32_t));
      (*word_assignments)[i][j] = w;
    }
      
    for (int j = 0; j < 128; ++j) {
      DescriptorType d;
      ifs.read((char*) &d, sizeof(DescriptorType));
      (*descriptors)(j, i) = d;
    }
  }

  ifs.close();
    
  return true;
}

// Save AffineSIFTKeypoints, their descriptors and visual word assignments
// from a binary file.
// Notice that the affine parameters are not altered! If the affine matrix needs
// to be inverted for consistency reasons, this needs to be done outside this
// function.
template <typename DescriptorType>
bool SaveAffineSIFTFeaturesAndMultipleNNWordAssignments(
     const std::string& filename, int num_nn,
     const std::vector<AffineSIFTKeypoint>& keypoints,
     const std::vector<std::vector<uint32_t> >& word_assignments,
     const Eigen::Matrix<DescriptorType, 128, Eigen::Dynamic>& descriptors) {
  std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    std::cerr << "ERROR: Cannot write the features to " << filename
              << std::endl;
  }

  uint32_t num_features;
  ofs.write((char*) &num_features, sizeof(uint32_t));

  if (num_features == 0) return true;

  for (uint32_t i = 0; i < num_features; ++i) {
    float x = keypoints[i].x[0];
    float y = keypoints[i].x[1];
    ofs.write((const char*) &x, sizeof(float));
    ofs.write((const char*) &y, sizeof(float));

    float a = keypoints[i].a;
    float b = keypoints[i].b;
    float c = keypoints[i].c;
    ofs.write((const char*) &a, sizeof(float));
    ofs.write((const char*) &b, sizeof(float));
    ofs.write((const char*) &c, sizeof(float));

    for (int j = 0; j < num_nn; ++j) {
      uint32_t w = word_assignments[i][j];
      ofs.write((const char*) &w, sizeof(uint32_t));
    }

    for (int j = 0; j < 128; ++j) {
      DescriptorType d = static_cast<DescriptorType>(descriptors(j, i));
      ofs.write((const char*) &d, sizeof(DescriptorType));
    }
  }

  ofs.close();

  return true;
}

}  // namespace geometric_burstiness

#endif  // GEOMETRIC_BURSTINESS_SRC_STRUCTURES_H_
