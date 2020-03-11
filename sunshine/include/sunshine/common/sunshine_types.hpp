//
// Created by stewart on 3/11/20.
//

#ifndef SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
#define SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP

namespace sunshine {

struct Phi {
  std::string id;
  int K = 0, V = 0;
  std::vector<std::vector<int>> counts = {};
  std::vector<int> topic_weights = {};

  explicit Phi()
        : id("") {}

  explicit Phi(std::string id)
        : id(std::move(id)) {}

  Phi(std::string id, uint32_t K, uint32_t V, std::vector<std::vector<int>> counts, std::vector<int> topic_weights)
        : id(std::move(id))
        , K(K)
        , V(V)
        , counts(std::move(counts))
        , topic_weights(std::move(topic_weights)) {
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

  void serialize(std::ostream &out) {
      if (!out.good()) throw std::logic_error("Output stream in invalid state");
      validate();
      constexpr int VERSION = INT_MIN + 1; // increment added constant whenever serialization format changes
      static_assert(VERSION < 0, "Version number must be negative!");
      out.write(reinterpret_cast<char const *>(&VERSION), sizeof(VERSION) / sizeof(char));
      out.write(reinterpret_cast<char *>(&K), sizeof(K) / sizeof(char));
      out.write(reinterpret_cast<char *>(&V), sizeof(V) / sizeof(char));
      out.write(reinterpret_cast<char const *>(topic_weights.data()), sizeof(decltype(topic_weights)::value_type) / sizeof(char) * K);
      for (auto word_dist : counts) {
          out.write(reinterpret_cast<char const *>(word_dist.data()), sizeof(decltype(word_dist)::value_type) / sizeof(char) * V);
      }
  }

  explicit Phi(std::istream &in) {
      if (!in.good()) throw std::logic_error("Input stream in invalid state");
      throw std::logic_error("Not yet implemented");
  }

//  explicit Phi(Phi &&other)
//        : id(other.id)
//        , K(other.K)
//        , V(other.V)
//        , counts(other.counts)
//        , topic_weights(other.topic_weights) {
//  }

  explicit Phi(std::istream &in, std::string id, int const &vocab_size)
        : id(std::move(id))
        , K(0)
        , V(vocab_size) {
      if (!in.good()) throw std::logic_error("Input stream in invalid state");
      std::vector<int> row(V, 0);
      while (!in.eof()) {
          in.read(reinterpret_cast<char *>(row.data()), sizeof(decltype(row)::value_type) / sizeof(char) * V);
          if (in.gcount() == 0) break;
          if (in.fail()) throw std::invalid_argument("Failed to read full row from data; wrong vocab size or corrupt data");

          K += 1;
          topic_weights.push_back(std::accumulate(row.begin(), row.end(), 0));
          counts.push_back(row);
      }
      assert(validate(false));
  }
};
}

#endif //SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
