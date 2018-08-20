#ifndef GEOMETRIC_BURSTINESS_SRC_INVERTED_FILE_H_
#define GEOMETRIC_BURSTINESS_SRC_INVERTED_FILE_H_

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "structures.h"

namespace geometric_burstiness {
// Implements an inverted file, including the ability to compute image scores
// and matches. The template parameter is the length of the binary vectors
// obtained via Hamming Embedding.
template <int N>
class InvertedFile {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  InvertedFile() : normalization_constant_(0.0), word_id_(-1) {
    static_assert(N % 8 == 0, "Dimensionality of projected space needs to"
                              " be a multiple of 8.");
    static_assert(N > 0, "Dimensionality of projected space needs to be > 0.");

    sigmas_per_component_.resize(256, N / 8);
    sigmas_per_component_.setZero();
    thresholds_.setZero();
  }

  typedef Eigen::Matrix<float, N, 1> ProjectedDesc;

  // Sets the id of the word this inverted file corresponds to.
  void SetWordId(int word_id) {
    word_id_ = word_id;
  }

  int WordId() const {
    return word_id_;
  }

  // Adds an inverted file entry. DOES NOT re-sort the inverted file!
  void AddEntry(const InvertedFileEntry<N>& entry) {
    entries_.push_back(entry);
  }

  size_t InvertedFileLength() const {
    return entries_.size();
  }

  // Adds an inverted file entry given a projected descriptor and its image
  // information stored in an inverted file entry. In particular, this function
  // generates the binary descriptor for the inverted file entry and then stores
  // the entry in the inverted file.
  void AddEntry(const ProjectedDesc& desc, const InvertedFileEntry<N>& entry) {
    InvertedFileEntry<N> e = entry;
    for (int i = 0; i < N; ++i) {
      e.descriptor[i] = desc[i] > thresholds_[i];
    }
    entries_.push_back(e);
  }

  // Returns a constant reference to the i-th inverted file entry.
  // DOES NOT check for bounds.
  const InvertedFileEntry<N>& GetIthEntry(int i) const {
    return entries_[i];
  }

  // Clears the inverted file.
  void ClearFile() {
    entries_.clear();
    normalization_constant_ = 0.0f;
    thresholds_.setZero();
    sigmas_per_component_.setZero();
    word_id_ = -1;
  }

  // Given a projected descriptor, returns the corresponding binary string.
  void ConvertToBinaryString(const ProjectedDesc& desc,
                             std::bitset<N>* bin_desc) const {
    for (int i = 0; i < N; ++i) {
      (*bin_desc)[i] = desc[i] > thresholds_[i];
    }
  }

  // Given a set of descriptors, learns the thresholds required for Hamming
  // embedding. Returns false if learning failed. Each column in descriptors
  // represents a single descriptor projected into the N dimensional space used
  // for Hamming embedding.
  bool LearnEmbeddingThreshold(
      const Eigen::Matrix<float, N, Eigen::Dynamic>& descriptors) {
    if (descriptors.cols() < 2) return false;
    int num_descriptors = static_cast<int>(descriptors.cols());
    std::vector<float> elements(num_descriptors);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < num_descriptors; ++j) {
        elements[j] = descriptors(i, j);
      }
      std::nth_element(elements.begin(), elements.begin() + num_descriptors / 2,
                       elements.end());
      thresholds_[i] = elements[num_descriptors / 2];
    }

    return true;
  }

  // Given the binary descriptors in the inverted file, learns the standard
  // deviations for each 8-dimensional subspace, as described in Sec. 3 of
  // Arandjelovic, Zisserman. DisLocation: Scalable descriptor distinctiveness
  // for location recognition. ACCV 2014.
  // The static_assertions in the constructor guarantee that N is
  // dividible by 8. Returns false if learning the sigmas failed.
  bool LearnSigmas(const HammingSpace<8>& hamming_space) {
    if (entries_.size() < 2u) return false;

    const int kNumSubSpaces = N / 8;
    const int num_descriptors = static_cast<int>(entries_.size());
    const double one_over_num_desc = 1.0 / static_cast<int>(num_descriptors);

    int index = 0;
    std::bitset<8> sub_string;
    // Later used to estimate the normalization constant.
    double sum_sigmas = 0.0;
    for (int s = 0; s < kNumSubSpaces; ++s, index += 8) {
      // As suggested in the paper, we first compute for each potential 8-bit
      // string the number of occurrences per component.
      std::vector<int> occurrences(256, 0);
      for (int i = 0; i < num_descriptors; ++i) {
        for (int j = 0; j < 8; ++j) {
          sub_string[j] = entries_[i].descriptor[index + j];
        }
        occurrences[sub_string.to_ulong()] += 1;
      }

      // Next, we compute the number of neighbors for each sub-descriptor that
      // has distance < N / (4 * kNumComponents) and use this to compute the
      // average number of neighbors p that is used to define \sigma_(c, b^s).
      const int kSigmaDefSub = static_cast<int>(N / (4 * kNumSubSpaces));
      int total_number_neighbors = 0;
      for (int i = 0; i < 256; ++i) {
        if (occurrences[i] <= 0) continue;
        // Counts the number of closer neighbors. Starts at -1 since the
        // descriptor will count itself.
        int num_nn = -1;
        for (int j = 0; j < 256; ++j) {
          if (hamming_space.distances_(j, i) < kSigmaDefSub) {
            num_nn += occurrences[j];
          }
        }
        total_number_neighbors += num_nn * occurrences[i];
      }
      int p = static_cast<int>(static_cast<double>(total_number_neighbors)
          / static_cast<double>(num_descriptors) + 0.5);

      // For each potential sub-string b^s, computes \sigma_s(c, b^s) such that
      // \sigma_s(c, b^s) is the distance to the p-th nearest neighbor.
      for (int i = 0; i < 256; ++i) {
        sigmas_per_component_(i, s) = static_cast<uint8_t>(0);

        int num_nn = -1;
        for (int j = 0; j < 256; ++j) {
          int j_closest_bit_vec = hamming_space.lists_(j, i);
          num_nn += occurrences[j_closest_bit_vec];
          if (num_nn >= p) {
            sigmas_per_component_(i, s) = static_cast<uint8_t>(hamming_space
                .distances_(j_closest_bit_vec, i));
            break;
          }
        }
      }

      for (int i = 0; i < 256; ++i) {
        sum_sigmas += static_cast<double>(sigmas_per_component_(i, s)
            * occurrences[i]) * one_over_num_desc;
      }
    }

    // Finally, we learn the normalization constant v_c that is used to
    // normalize \sigma(c,b). To do so, we estimate the mean sigma for all
    // descriptors.
    float mean_sigma = static_cast<float>(sum_sigmas);

    normalization_constant_ = static_cast<float>(N) * 0.25f - mean_sigma;

    return true;
  }

  // Given a query feature with its id and binary descriptor representation,
  // and a vote weighting function, performs inverted file scoring. Returns
  // a list of scores and matches.
  void PerformVoting(
      const QueryDescriptor<N>& query_desc,
                     const VoteWeightingFunction<N>& f, const float idf_weigth,
                     std::vector<ImageScore>* scores_and_matches) const {
    float squared_idf_weight = idf_weigth * idf_weigth;
    std::vector<ImageScore>& sam_ref = *scores_and_matches;

    int num_entries = static_cast<int>(entries_.size());
    if (num_entries == 0) return;

    // Thresholds the query descriptor to obtain a binary string.
    std::bitset<N> q;
    for (int i = 0; i < N; ++i) {
      q[i] = query_desc.proj_desc[i] > thresholds_[i];
    }

    std::vector<float> counts(N, 0.0f);
    for (int i = 0; i < num_entries; ++i) {
      for (int j = 0; j < N; ++j) {
        counts[j] += entries_[i].descriptor[j]? 1.0f : 0.0f;
      }
    }

    ImageScore current_score;
    current_score.image_id = entries_[0].image_id;
    current_score.voting_weight = 0.0f;
    current_score.matches.clear();
    int index = 0;
    const int kNumSubSpaces = N / 8;  // Constructor ensures N / 8 is integer.
    std::bitset<8> sub_string;

    size_t max_hamming_distance = 32u;
    for (size_t i = 0; i < query_desc.relevant_word_ids.size(); ++i) {
      if (query_desc.relevant_word_ids[i] == word_id_) {
        max_hamming_distance = std::min(
            query_desc.max_hamming_distance_per_word[i], max_hamming_distance);
        break;
      }
    }

    int num_skipped = 0;
    for (int i = 0; i <= num_entries; ++i) {
      if ((i == num_entries)
          || (current_score.image_id < entries_[i].image_id)) {
        if (!current_score.matches.empty()) {
          // Finalizes the voting since we now know how many features from
          // the database image match the current image feature. This is
          // required to perform burstiness normalizaation (cf. Eqn. 2 in
          // Arandjelovic, Zisserman. DisLocation: Scalable descriptor
          // distinctiveness for location recognition. ACCV 2014.
          // Notice that the weight from the descriptor matching is already
          // accumulated in current_score.voting_weight, i.e., we only need
          // to apply the burstiness weighting.
          float sqrt_num_matches = static_cast<float>(std::sqrt(
              static_cast<double>(current_score.matches.size())));
          // Notice that the idf weight is incorporated by the inverted index
          // rather than the inverted file (due to me not wanting to update
          // this part before the deadline).
          // TODO(sattler): Move all idf weighting inside this function.
          current_score.voting_weight /= sqrt_num_matches;
          sam_ref.push_back(current_score);
        }

        current_score.image_id = entries_[i].image_id;
        current_score.voting_weight = 0.0f;
        current_score.matches.clear();
      }

      // Computes the Hamming distance.
      size_t hamming_dist = (q ^ entries_[i].descriptor).count();

      if (hamming_dist > max_hamming_distance) {
        ++num_skipped;
        continue;
      }

      // Adaptively compute the sigma as proposed in the paper.
      int sigma = 0;
      index = 0;
      // Faster implementation to get the substrings.
      uint64_t val_bin_string = entries_[i].descriptor.to_ulong();
      for (int s = 0; s < 8; ++s) {
        uint64_t sub_index = val_bin_string & 255;
        val_bin_string = val_bin_string >> 8;
        sigma += sigmas_per_component_(sub_index, s);
      }

      int sigma_final = static_cast<int>(static_cast<float>(sigma)
          + normalization_constant_ + 0.5f);
      // Avoids accessing the look-up table out of bounds.
      sigma_final = std::max(0, sigma_final);
      sigma_final = std::min(sigma_final, N);

      // Based on the Hamming distance, computes the weight for the vote, and
      // finally creates a match if necessary.
      float weight = f.GetWeight(
          hamming_dist,
          std::min(sigma_final, static_cast<int>(max_hamming_distance)));

      if (weight > 0.0f) {
        current_score.voting_weight += weight;
        Match2D2D m(query_desc.feature_id, word_id_, i,
                    weight * squared_idf_weight);
        current_score.matches.push_back(m);
      } else {
        ++num_skipped;
      }
    }
  }

  // Sorts the inverted file in ascending order of image ids. This is required
  // for efficient scoring.
  void SortInvertedFile() {
    std::sort(entries_.begin(), entries_.end(), CmpInvertedFileEntries<N>);
  }

  // Writes the inverted file into a binary file.
  bool SaveInvertedFile(std::ofstream* ofs) const {
    if (!ofs->is_open()) {
      std::cerr << "ERROR: Stream not open for writing " << std::endl;
      return false;
    }

    int32_t word_id = static_cast<int32_t>(word_id_);
    ofs->write((const char*) &word_id, sizeof(int32_t));

    ofs->write((const char*) &normalization_constant_, sizeof(float));

    int32_t N_t = static_cast<int32_t>(N);
    ofs->write((const char*) &N_t, sizeof(int32_t));

    for (int i = 0; i < N; ++i) {
      float val = thresholds_[i];
      ofs->write((const char*) &val, sizeof(float));
    }

    for (int j = 0; j < (N / 8); ++j) {
      for (int i = 0; i < 256; ++i) {
       uint8_t val = sigmas_per_component_(i, j);
       ofs->write((const char*) &val, sizeof(uint8_t));
      }
    }

    uint64_t num_entries = static_cast<uint64_t>(entries_.size());
    ofs->write((const char*) &num_entries, sizeof(uint64_t));

    for (uint64_t i = 0; i < num_entries; ++i) {
      if (!WriteInvertedFileEntryFromBinary(entries_[i], ofs)) {
        return false;
      }
    }

    return true;
  }

  // Reads the inverted file from a binary file.
  bool ReadInvertedFile(std::ifstream* ifs) {
    if (!ifs->is_open()) {
      std::cerr << "ERROR: Stream not open for reading " << std::endl;
      return false;
    }

    int32_t word_id = 0;
    ifs->read((char*) &word_id, sizeof(int32_t));
    word_id_ = static_cast<int>(word_id);

    ifs->read((char*) &normalization_constant_, sizeof(float));

    int32_t N_t = static_cast<int32_t>(N);
    ifs->read((char*) &N_t, sizeof(int32_t));
    if (static_cast<int>(N_t) != N) {
      std::cerr << "ERROR: The length of the binary strings should be " << N
                << " but is " << N_t << ". The indices are not compatible!"
                << std::endl;
      return false;
    }

    for (int i = 0; i < N; ++i) {
      float val = 0.0f;
      ifs->read((char*) &val, sizeof(float));
      thresholds_[i] = val;
    }

    for (int j = 0; j < (N / 8); ++j) {
      for (int i = 0; i < 256; ++i) {
        uint8_t val = static_cast<uint8_t>(0);
        ifs->read((char*) &val, sizeof(uint8_t));
        sigmas_per_component_(i, j) = val;
      }
    }

    uint64_t num_entries = 0u;
    ifs->read((char*) &num_entries, sizeof(uint64_t));
    entries_.resize(num_entries);

    for (uint64_t i = 0; i < num_entries; ++i) {
      InvertedFileEntry<N> entry;
      if (!ReadInvertedFileEntryFromBinary(ifs, &entry)) {
        std::cout << "ERROR: Could not read entry " << i << std::endl;
        return false;
      }
      entries_[i] = entry;
    }

    return true;
  }
  
  void GetImageIdsInFile(std::unordered_set<int>* ids) const {
    for (const InvertedFileEntry<N>& entry : entries_) {
      ids->insert(entry.image_id);
    }
  }
  
  void GetEntriesForImage(int image_id,
                          std::vector<InvertedFileEntry<N> >* entries) const {
    for (const InvertedFileEntry<N>& entry : entries_) {
      if (entry.image_id != image_id) continue;
      entries->push_back(entry);
    }
  }

  // Returns the idf weight for this inverted file.
  float GetIDFWeight(int num_total_images) const {
    if (entries_.empty()) return 0.0f;

    std::unordered_set<int> images_in_file;
    GetImageIdsInFile(&images_in_file);
    float idf_weight = static_cast<float>(std::log(
        static_cast<double>(num_total_images)
            / static_cast<double>(images_in_file.size())));
    return idf_weight;
  }

  // For each image in the inverted file, computes the self-similarity of each
  // image in the file (at least the part caused by this word) and adds the
  // weight to the entry corresponding to that image. This function is useful
  // to determine the normalization factor for each image that is used during
  // retrieval.
  bool DetermineDBImageSelfSimilarities(
      const VoteWeightingFunction<N>& f,
      float idf_weight,
      std::vector<double>* self_similarity_scores) const {
    int num_entries = static_cast<int>(entries_.size());

    if (num_entries == 0) return true;

    const double idf_squared = static_cast<double>(idf_weight)
        * static_cast<double>(idf_weight);

    int total_num_images = static_cast<int>(self_similarity_scores->size());
    std::vector<double>& score_ref = *self_similarity_scores;
    const int kNumSubSpaces = N / 8;
    std::bitset<8> sub_string;

    int first_index_image = 0;
    int current_image_id = entries_[0].image_id;
    int score_size = score_ref.size();
    std::vector<double> num_vws(score_size,0.0);

    for (int i = 0; i < num_entries; ++i) {
      current_image_id = entries_[i].image_id;
      num_vws[current_image_id] += 1;
    }       
    for (int i = 0; i < score_size; ++i) {         
      score_ref[i] += num_vws[i]*num_vws[i]*idf_squared;
    }

    num_vws.clear();
    return true;
  }

  void PrintBinaryDescriptorStats() const {

    std::vector<int> count_ones(N, 0);
    for (const InvertedFileEntry<N>& entry : entries_) {
      for (int i = 0; i < N; ++i) {
        if (entry.descriptor[i]) ++count_ones[i];
      }
    }
    double num_entries = static_cast<double>(entries_.size());
    bool print = false;
    for (int i = 0; i < N; ++i) {
      double val = static_cast<double>(count_ones[i]) / num_entries;
      if (fabs(val - 0.5) > 0.1) print = true;
    }
    if (print) {
      std::cout << word_id_ << " | ";
      std::cout << word_id_ << " -- " << normalization_constant_ << " "
                << entries_.size() << std::endl;
      for (int i = 0; i < N; ++i) {
        double val = static_cast<double>(count_ones[i]) / num_entries;
        std::cout << val << " ";
      }
      std::cout << std::endl;
    }
  }

  void SetHammingThresholds(const ProjectedDesc& thresholds) {
    thresholds_ = thresholds;
  }

  void SetSigmasRelja(const Eigen::Matrix<float, 256, Eigen::Dynamic>& sigmas) {
    sigmas_per_component_relja_ = sigmas;
  }

 private:
  // The entries of the inverted file system.
  std::vector<InvertedFileEntry<N> > entries_;
  // The normalization constant used to ensure that the average standard
  // deviation used for computing the weight of a vote is N / 4;
  float normalization_constant_;
  // For each possible binary substring of length 8, stores the value of
  // \sigma_s(c, b^{(s)}) as described in the paper.
  Eigen::Matrix<uint8_t, 256, Eigen::Dynamic> sigmas_per_component_;
  Eigen::Matrix<float, 256, Eigen::Dynamic> sigmas_per_component_relja_;
  // The thresholds used for Hamming embedding.
  ProjectedDesc thresholds_;
  // The id of the word, i.e., the index of this inverted file in the inverted
  // file system.
  int word_id_;
};

}  // namespace geometric_burstiness

#endif  // GEOMETRIC_BURSTINESS_SRC_INVERTED_FILE_H_
