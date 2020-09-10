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
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>

namespace sunshine {

struct Phi {
  constexpr static uint32_t VERSION = 0; // increment added constant whenever serialization format changes
  std::string id;
  int K = 0, V = 0;
  std::vector<std::vector<int>> counts = {};
  std::vector<int> topic_weights = {};
  bool validated = false;

  explicit Phi() = default;

  explicit Phi(std::string id)
          : id(std::move(id)) {}

  explicit Phi(std::string id, uint32_t K, uint32_t V, std::vector<std::vector<int>> counts, std::vector<int> topic_weights)
          : id(std::move(id)), K(K), V(V), counts(std::move(counts)), topic_weights(std::move(topic_weights)) {
  }

  void remove_unused() {
      throw std::logic_error("Needs re-thinking -- this will mess up map topic indices");
      auto const starting_size = topic_weights.size();
      for (size_t i = starting_size; i > 0; --i) {
          auto const idx = i - 1;
          if (topic_weights[idx] == 0) {
              topic_weights.erase(topic_weights.begin() + idx);
              counts.erase(counts.begin() + idx);
              K -= 1;
          }
      }
  }

  bool validate(bool verbose = true) {
      bool flag = true;
      if (K != counts.size() || K != topic_weights.size()) {
          if (verbose) {
              std::cerr << "Mismatch between K=" << K << ", counts.size()=" << counts.size() << ", and topic_weights.size()="
                        << topic_weights.size() << std::endl;
          }
          flag = false;
          K = counts.size();
          topic_weights.resize(K, 0);
      }
      for (auto k = 0; k < K; ++k) {
          assert(k == 0 || V == counts[k].size());
          if (V != counts[k].size()) {
              if (verbose) std::cerr << "Mismatch between V=" << V << ", counts[k].size()=" << counts[k].size() << std::endl;
              flag = false;
              V = counts[k].size();
          }
          int weight = 0;
          for (auto w = 0; w < V; ++w) weight += counts[k][w];
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

  Phi(Phi const &other) = default;

  Phi(Phi &&other) noexcept = default;

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
  }

  template<typename Archive>
  void load(Archive &ar, const unsigned int version) {
      if (version != VERSION) throw std::invalid_argument("Unexpected serialization format.");

      ar & id;
      ar & K;
      ar & V;
      ar & topic_weights;
      ar & counts;
      assert(this->validate(false));
      this->validated = true;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER(); // Required for separate load/save functions
};
}
BOOST_CLASS_VERSION(sunshine::Phi, sunshine::Phi::VERSION);

#endif //SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
