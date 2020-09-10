//
// Created by stewart on 3/11/20.
//

#ifndef SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
#define SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP

#include <string>
#include <vector>
#include <iostream>
#include <numeric>

namespace sunshine {

struct Phi {
  constexpr static int32_t VERSION = INT_MIN + 3; // increment added constant whenever serialization format changes
  std::string id;
  int K = 0, V = 0;
  std::vector<std::vector<int>> counts = {};
  std::vector<int> topic_weights = {};
  bool validated = false;

  explicit Phi() = default;

  explicit Phi(std::string id)
        : id(std::move(id)) {}

  explicit Phi(std::string id, uint32_t K, uint32_t V, std::vector<std::vector<int>> counts, std::vector<int> topic_weights)
        : id(std::move(id))
        , K(K)
        , V(V)
        , counts(std::move(counts))
        , topic_weights(std::move(topic_weights)) {
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

  template <typename StreamType>
  void serialize(StreamType &out) const {
//      if (!out.good()) throw std::invalid_argument("Output stream in invalid state");
      if (id.size() > std::numeric_limits<uint8_t>::max()) throw std::invalid_argument("Identifier " + id + " is too long to serialize");
      if (!validated) throw std::logic_error("Must validate Phi object before serializing");
      static_assert(VERSION < 0, "Version number must be negative!"); // just to keep it looking distinctive
      out.write(reinterpret_cast<char const *>(&VERSION), sizeof(VERSION));

      uint8_t const id_len = id.size();
      out.write(reinterpret_cast<char const *>(&id_len), sizeof(uint8_t));
      out.write(reinterpret_cast<char const *>(id.data()), id_len);
      out.write(reinterpret_cast<char const *>(&K), sizeof(K));
      out.write(reinterpret_cast<char const *>(&V), sizeof(V));
      out.write(reinterpret_cast<char const *>(topic_weights.data()), sizeof(decltype(topic_weights)::value_type) * K);
      for (auto const& word_dist : counts) {
          out.write(reinterpret_cast<char const *>(word_dist.data()), sizeof(std::remove_reference_t<decltype(word_dist)>::value_type) * V);
      }
      out.flush();
  }

  Phi(Phi const& other) = default;
  Phi(Phi&& other) noexcept = default;

  template <typename StreamType>
  static Phi deserialize(StreamType &in) {
      Phi phi;
//      if (!in.good()) throw std::invalid_argument("Input stream in invalid state");
      std::remove_const_t<decltype(VERSION)> version;
      in.read(reinterpret_cast<char *>(&version), sizeof(version));
      if (version != VERSION) throw std::invalid_argument("Unexpected serialization format.");

      uint8_t id_len;
      std::vector<char> id_chars;
      in.read(reinterpret_cast<char *>(&id_len), sizeof(uint8_t));
      id_chars.resize(id_len, 0);
      in.read(id_chars.data(), id_len);
      phi.id = std::string(id_chars.begin(), id_chars.end());
      in.read(reinterpret_cast<char *>(&phi.K), sizeof(K));
      in.read(reinterpret_cast<char *>(&phi.V), sizeof(V));
      phi.topic_weights.resize(phi.K, 0);
      in.read(reinterpret_cast<char *>(phi.topic_weights.data()), sizeof(decltype(topic_weights)::value_type) * phi.K);
      decltype(counts)::value_type row(phi.V, 0);
      for (auto i = 0; i < phi.K; ++i) {
          in.read(reinterpret_cast<char *>(row.data()), sizeof(decltype(row)::value_type) * phi.V);
          if (in.gcount() == 0 || in.fail()) throw std::invalid_argument("Failed to read full row from data; wrong vocab size or corrupt data");
          phi.counts.push_back(row);
      }
      assert(phi.validate(false));
      return phi;
  }
};
}

#endif //SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
