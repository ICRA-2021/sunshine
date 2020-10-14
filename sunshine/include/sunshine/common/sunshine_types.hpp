//
// Created by stewart on 3/11/20.
//

#ifndef SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
#define SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP

#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/array.hpp>
#include "sparse_vector.hpp"

namespace sunshine {

struct Phi {
  constexpr static uint32_t VERSION = 2; // increment added constant whenever serialization format changes
  std::string id;
  int K = 0, V = 0;
  std::vector<sparse_vector<int, uint32_t>> counts = {};
  std::vector<int> topic_weights = {};
  uint64_t cell_refines = 0, word_refines = 0;
  bool validated = false;

  explicit Phi() = default;

  explicit Phi(std::string id)
          : id(std::move(id)) {}

  explicit Phi(std::string id, uint32_t K, uint32_t V, std::vector<std::vector<int>> counts, std::vector<int> topic_weights, uint64_t cell_refines = 0, uint64_t word_refines = 0)
          : id(std::move(id)), K(K), V(V), counts({counts.begin(), counts.end()}), topic_weights(std::move(topic_weights)), cell_refines(cell_refines), word_refines(word_refines) {
  }

  explicit operator std::vector<std::vector<int>>() const {
      return {counts.begin(), counts.end()};
  }

  void remove_unused() {
      throw std::logic_error("Needs re-thinking -- this will mess up map topic indices");
//      auto const starting_size = topic_weights.size();
//      for (size_t i = starting_size; i > 0; --i) {
//          auto const idx = i - 1;
//          if (topic_weights[idx] == 0) {
//              topic_weights.erase(topic_weights.begin() + idx);
//              counts.erase(counts.begin() + idx);
//              K -= 1;
//          }
//      }
  }

  bool validate(bool verbose = true) {
      bool flag = true;
      if (static_cast<uint32_t>(K) != counts.size() || static_cast<uint32_t>(K) != topic_weights.size()) {
          if (verbose) {
              std::cerr << "Mismatch between K=" << K << ", counts.size()=" << counts.size() << ", and topic_weights.size()="
                        << topic_weights.size() << std::endl;
          }
          flag = false;
          K = counts.size();
          topic_weights.resize(K, 0);
      }
      for (auto k = 0; k < K; ++k) {
          assert(k == 0 || static_cast<uint32_t>(V) == counts[k].size());
          if (static_cast<uint32_t>(V) != counts[k].size()) {
              if (verbose) std::cerr << "Mismatch between V=" << V << ", counts[k].size()=" << counts[k].size() << std::endl;
              flag = false;
              V = counts[k].size();
          }
          int const weight = counts[k].accumulate(0);
          if (weight != topic_weights[k]) {
              if (verbose) {
                  std::cerr << "Mismatch between computed topic weight " << weight << " and topic_weights[k]=" << topic_weights[k]
                            << std::endl;
              }
              flag = false;
              topic_weights[k] = weight;
          }
      }
      validated = true;
      return flag;
  }

  template<typename Archive>
  void save(Archive &ar, const unsigned int version) const {
//      if (!out.good()) throw std::invalid_argument("Output stream in invalid state");
      if (version != VERSION) throw std::logic_error("Unexpected serialization version.");
      if (!validated) throw std::logic_error("Must validate Phi object before serializing");

      ar & id;
      ar & K;
      ar & V;
      ar & topic_weights;
      ar & counts;
      ar & cell_refines;
      ar & word_refines;
  }

  template<typename Archive>
  void load(Archive &ar, const unsigned int version) {
      ar & id;
      ar & K;
      ar & V;
      ar & topic_weights;
      if (version < 1) {
          std::vector<std::vector<int>> vec_counts;
          ar & vec_counts;
          counts = {vec_counts.begin(), vec_counts.end()};
      } else {
          ar & counts;
      }
      if (version >= 2) {
          ar & cell_refines;
          ar & word_refines;
      }
      assert(this->validate(false));
      this->validated = true;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER(); // Required for separate load/save functions

  size_t bytesSize() const {
      auto const headerSize = id.size() + sizeof(K) + sizeof(V);
      auto const weightSize = sizeof(topic_weights) + sizeof(decltype(topic_weights)::value_type) * topic_weights.capacity();
      auto const countVecSize = sizeof(counts) + sizeof(decltype(counts)::value_type) * topic_weights.capacity();
      size_t countsSize = 0;
      for (auto const& c : counts) {
          countsSize += c.bytesSize();
      }
      return headerSize + weightSize + countVecSize + countsSize;
  }
};
}
BOOST_CLASS_VERSION(sunshine::Phi, sunshine::Phi::VERSION);

#endif //SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
