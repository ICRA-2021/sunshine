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
  constexpr static int32_t VERSION = INT_MIN + 2; // increment added constant whenever serialization format changes
  std::string id;
  int K = 0, V = 0;
  std::vector<std::vector<int>> counts = {};
  std::vector<int> topic_weights = {};

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
      return flag;
  }

  void serialize(std::ostream &out) const {
      if (!out.good()) throw std::invalid_argument("Output stream in invalid state");
      if (id.size() > std::numeric_limits<uint8_t>::max()) throw std::invalid_argument("Identifier " + id + " is too long to serialize");
      static_assert(VERSION < 0, "Version number must be negative!"); // just to keep it looking distinctive
      out.write(reinterpret_cast<char const *>(&VERSION), sizeof(VERSION));
      uint8_t const id_len = id.size();
      out.write(reinterpret_cast<char const *>(&id_len), sizeof(uint8_t));
      out.write(reinterpret_cast<char const *>(id.data()), id_len);
      out.write(reinterpret_cast<char const *>(&K), sizeof(K));
      out.write(reinterpret_cast<char const *>(&V), sizeof(V));
      out.write(reinterpret_cast<char const *>(topic_weights.data()), sizeof(decltype(topic_weights)::value_type) * K);
      for (auto word_dist : counts) {
          out.write(reinterpret_cast<char const *>(word_dist.data()), sizeof(decltype(word_dist)::value_type) * V);
      }
  }

//  explicit Phi(Phi &&other)
//        : id(other.id)
//        , K(other.K)
//        , V(other.V)
//        , counts(other.counts)
//        , topic_weights(other.topic_weights) {
//  }

  explicit Phi(std::istream &in) {
      if (!in.good()) throw std::invalid_argument("Input stream in invalid state");
      int32_t version;
      in.read(reinterpret_cast<char *>(&version), sizeof(version));
      if (version != VERSION) throw std::invalid_argument("Unexpected serialization format.");
      uint8_t id_len;
      std::vector<char> id_chars;
      in.read(reinterpret_cast<char *>(&id_len), sizeof(uint8_t));
      id_chars.resize(id_len, 0);
      in.read(id_chars.data(), id_len);
      id = std::string(id_chars.begin(), id_chars.end());
      in.read(reinterpret_cast<char *>(&K), sizeof(K));
      in.read(reinterpret_cast<char *>(&V), sizeof(V));
      topic_weights.resize(K, 0);
      in.read(reinterpret_cast<char *>(topic_weights.data()), sizeof(decltype(topic_weights)::value_type) * K);
      decltype(counts)::value_type row(V, 0);
      for (auto i = 0; i < K; ++i) {
          in.read(reinterpret_cast<char *>(row.data()), sizeof(decltype(row)::value_type) * V);
          if (in.gcount() == 0 || in.fail()) throw std::invalid_argument("Failed to read full row from data; wrong vocab size or corrupt data");
          counts.push_back(row);
      }
      assert(in.eof());
      assert(validate(false));
  }
};
}

#endif //SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
