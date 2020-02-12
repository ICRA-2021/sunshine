#ifndef SUNSHINE_PROJECT_ADROST_UTILS_HPP
#define SUNSHINE_PROJECT_ADROST_UTILS_HPP

#include <cmath>
#include <munkres/munkres.h>
#include <clear/MultiwayMatcher.hpp>
#include <Eigen/Core>
#include <numeric>
// #include "adapters/boostmatrixadapter.h"

struct Phi {
  std::string id;
  int K = 0, V = 0;
  std::vector<std::vector<int>> counts = {};
  std::vector<int> topic_weights = {};

  explicit Phi(std::string id)
        : id(std::move(id)) {}

  Phi(std::string id, int K, int V, std::vector<std::vector<int>> counts, std::vector<int> topic_weights)
        : id(std::move(id))
        , K(K)
        , V(V)
        , counts(std::move(counts))
        , topic_weights(std::move(topic_weights)) {
  }

  bool validate() {
      bool flag = true;
      if (K != counts.size() || K != topic_weights.size()) {
          ROS_WARN("Mismatch between K=%d, counts.size()=%lu, and topic_weights.size()=%lu", K, counts.size(), topic_weights.size());
          flag = false;
          K = counts.size();
          topic_weights.resize(K, 0);
      }
      for (auto k = 0; k < K; ++k) {
          assert(k == 0 || V == counts[k].size());
          if (V != counts[k].size()) {
              ROS_WARN("Mismatch between V=%d, counts[k].size()=%lu", K, counts[k].size());
              flag = false;
              V = counts[k].size();
          }
          int weight = 0;
          for (auto w = 0; w < V; ++w) weight += counts[k][w];
          if (weight != topic_weights[k]) {
              ROS_WARN("Mismatch between computed topic weight %d and topic_weights[k]=%d", weight, topic_weights[k]);
              flag = false;
              topic_weights[k] = weight;
          }
      }
      return flag;
  }
};

struct match_results {
  int num_unique = -1;
  std::vector<std::vector<int>> lifting = {};
  std::vector<double> ssd = {};
  std::vector<double> matched_ssd = {};
};

/**
 * Computes the squared euclidean distance between two vectors, v and w
 * @param v the first vector<float>
 * @param w the second vector<float>
 * @returns float - the squared eucl. distance between v and w
**/
template<typename T>
double normed_dist_sq(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {

    assert (v.size() == w.size());
    double distance_sq = 0.0;
    double diff;

    if (scale_v == 0 && scale_w == 0) { return 0.; }
    else if (scale_v == 0 || scale_w == 0) { return 1.; }

    double const invscale_v = 1. / scale_v, invscale_w = 1. / scale_w;

    for (auto i = 0ul; i < v.size(); ++i) {
        // diff = 10000.0f*v[i] - 10000.0f*w[i];
        diff = (v[i] * invscale_v) - (w[i] * invscale_w);
        distance_sq += (diff * diff);
    }

    return distance_sq;
}

template<typename T>
double kl_div(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    if (scale_v == 0 && scale_w == 0) { return 0.; }
    else if (scale_v == 0 || scale_w == 0) { return std::numeric_limits<double>::infinity(); }

    assert (v.size() == w.size());
    // TODO validate this code
    return std::inner_product(v.begin(),
                              v.end(),
                              w.begin(),
                              0,
                              [](double const &sum, double const &prod) { return sum + prod; },
                              [=](T const &left, T const &right) {
                                  return (left / scale_v) * std::log2(static_cast<double>(left * scale_w) / (right * scale_v));
                              });
}

template<typename T>
double symmetric_kl_div(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    return kl_div(v, w, scale_v, scale_w) + kl_div(w, v, scale_w, scale_v);
}

template<typename T>
double jensen_shannon_div(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    assert (v.size() == w.size());
    std::vector<double> m;
    std::transform(v.begin(), v.end(), w.begin(), std::back_inserter(m), [](T const &left, T const &right) { return (left + right) / 2.; });
    auto const norm_m = (scale_v + scale_w) / 2.;
    // TODO validate this code
    return (kl_div(std::vector<double>(v.begin(), v.end()), m, scale_v, norm_m)
          + kl_div(std::vector<double>(w.begin(), w.end()), m, scale_w, norm_m)) / 2.;
}

/**
 * Computes the cosine similarity two vectors, v and w
 * @param v the first vector<float>
 * @param w the second vector<float>
**/
template<typename T>
double cosine_similarity(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    if (scale_v == 0 && scale_w == 0) { return 1.; }
    else if (scale_v == 0 || scale_w == 0) { return 0.; }

    auto const add_square = [](double const &sum, T const &next) { return sum + (static_cast<double>(next) * next); };
    double const norm_v = std::sqrt(std::accumulate(v.begin(), v.end(), 0., add_square));
    double const norm_w = std::sqrt(std::accumulate(w.begin(), w.end(), 0., add_square));
    double const dot_p = std::inner_product(v.begin(),
                                            v.end(),
                                            w.begin(),
                                            0.0,
                                            [](double const &sum, double const &prod) { return sum + prod; },
                                            [](T const &left, T const &right) {
                                                return static_cast<double>(left) * right;
                                            }); // coerce ints to doubles

    assert (v.size() == w.size());
    return dot_p / (norm_v * norm_w);
}

/**
 * Computes the Bhattacharyya coefficient between two vectors, v and w
 * @param v the first vector<float>
 * @param w the second vector<float>
**/
template<typename T>
double bhattacharyya_coeff(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    if (scale_v == 0 && scale_w == 0) { return 1.; }
    else if (scale_v == 0 || scale_w == 0) { return 0.; }

    assert (v.size() == w.size());
    double const rt_scale = std::sqrt(scale_v * scale_w);
    double sum = 0.0;
    for (auto i = 0ul; i < v.size(); ++i) {
        if (v[i] > 0 && w[i] > 0) {
            sum += std::sqrt(static_cast<double>(v[i]) * w[i]) / rt_scale;
        }
    }
    return sum;
}

/**
 * Permutes the rows of a
 * @param vector a, to be permuted
 * @param vector<int> perm. perm_i corresponds to the unique destination row of a_i
 *	so len(perm) == len(a) and max(perm) == len(a) - 1
 */
template<typename T>
std::vector<T> permute(std::vector<T> a, std::vector<int> perm) {
    std::vector<T> permuted(a);
    for (int i = 0; i < a.size(); ++i) {
        permuted[perm[i]] = a[i];
    }

    return permuted;
}

/**
 * Merges two topic-word count matrices into a global topic-word matrix by id
 * @param nZW_1 the first topic-word count matrix
 * @param nZW_2 the second topic-word count matrix
 * @param nZW_global the global topic-word count matrix to update
 * @param K the number of topics
 * @param V the size of the vocabulary
**/
std::vector<std::vector<int>> merge_by_id(std::vector<std::vector<int>> nZW_1,
                                          std::vector<std::vector<int>> nZW_2,
                                          std::vector<std::vector<int>> nZW_global,
                                          int K,
                                          int V) {
    for (int k = 0; k < K; ++k) {
        for (int v = 0; v < V; ++v) {
            int prev = nZW_global[k][v];
            //for (int r = 0; r < N_robots; ++r) {
            nZW_global[k][v] = nZW_global[k][v] + (nZW_1[k][v] - prev);
            nZW_global[k][v] = nZW_global[k][v] + (nZW_2[k][v] - prev);
            //}
        }
    }
    return nZW_global;
}

/**
 * Computes the pairwise squared eucl. distances between rows of matrices a and b
 * @param a matrix of floats
 * @param b matrix of floats
 * @returns matrix of squared distances, size (num rows a) x (num rows b)
**/
template<typename T>
std::vector<std::vector<double>> pairwise_distance_sq(std::vector<std::vector<T>> const &a,
                                                      std::vector<std::vector<T>> const &b,
                                                      std::vector<T> const &norm_a = {},
                                                      std::vector<T> const &norm_b = {}) {
    std::vector<std::vector<double>> pd(a.size(), std::vector<double>(b.size(), 0.0));
    for (auto i = 0ul; i < a.size(); ++i) {
        for (auto j = 0ul; j < b.size(); ++j) {
            pd[i][j] = normed_dist_sq(a[i],
                                      b[j],
                                      (norm_a.empty())
                                      ? 1
                                      : norm_a[i],
                                      (norm_b.empty())
                                      ? 1.
                                      : norm_b[j]);
        }
    }
    return pd;
}

/**
 * Given assignment matrix, returns correct destination of robot 2 topics
 *
**/
std::vector<int> get_permutation(std::vector<std::vector<int>> assignment, int *next_id = nullptr) {
    // Default perm to -1 so things fail catastrophically if all doesn't go well
    std::vector<int> perm(assignment[0].size(), -1);
    for (int j = 0; j < assignment[0].size(); ++j) {
        bool matched = false;
        for (int i = 0; i < assignment.size(); ++i) {
            if (assignment[i][j] == 0) {
                perm[j] = i;
                matched = true;
                break;
            }
        }
        if (!matched && next_id != nullptr) perm[j] = (*next_id)++;
    }
    return perm;
}

/**
 * Merges two topic-word count matrices into a global topic-word matrix by similarity
 * @param nZW_1 the first topic-word count matrix
 * @param phi_1 the first topic-word distribution matrix
 * @param nZW_2 the second topic-word count matrix
 * @param phi_2 the second topic-word distribution matrix
 * @param nZW_global the global topic-word count matrix to update
 * @param K the number of topics
 * @param V the size of the vocabulary
**/

/**
 * Placeholder to keep track of nZW global and robot 2 permutation
**/

match_results id_matching(std::vector<Phi> const &topic_models) {
    match_results results = {};
    if (topic_models.empty()) return results;

    auto const &left = topic_models[0].counts;
    auto const &left_weights = topic_models[0].topic_weights;

    results.num_unique = left_weights.size();
    results.lifting.emplace_back();
    for (auto i = 0ul; i < left_weights.size(); ++i) {
        results.lifting[0].push_back(i);
    }

    results.ssd = std::vector<double>(1, 0); // SSD with self is 0
    results.matched_ssd = std::vector<double>(1, 0); // SSD with self is 0

    for (auto i = 1ul; i < topic_models.size(); ++i) {
        auto const &right = topic_models[i].counts;
        auto const &right_weights = topic_models[i].topic_weights;
        Matrix<double> matrix(left_weights.size(), right_weights.size());

        std::vector<std::vector<double>> pd_sq = pairwise_distance_sq(left, right, left_weights, right_weights);
        std::vector<std::vector<int>> assignment(left_weights.size(), std::vector<int>(right_weights.size(), -1));

        results.ssd.push_back(0);
        for (uint64_t row = 0; row < left_weights.size(); row++) {
            assignment[row][row] = 0;
            results.ssd.back() += pd_sq[row][row];
        }
        results.lifting.push_back(get_permutation(assignment, &results.num_unique));
    }
    results.matched_ssd = results.ssd; // identity matching
    return results;
}

match_results sequential_hungarian_matching(std::vector<Phi> const &topic_models) {
    match_results results = {};
    if (topic_models.empty()) return results;

    auto const &left = topic_models[0].counts;
    auto const &left_weights = topic_models[0].topic_weights;

    results.num_unique = left_weights.size();
    results.lifting.emplace_back();
    for (auto i = 0ul; i < left_weights.size(); ++i) {
        results.lifting[0].push_back(i);
    }
    results.ssd = std::vector<double>(1, 0); // SSD with self is 0
    results.matched_ssd = std::vector<double>(1, 0); // SSD with self is 0

    for (auto i = 1ul; i < topic_models.size(); ++i) {
        auto const &right = topic_models[i].counts;
        auto const &right_weights = topic_models[i].topic_weights;
//        ROS_WARN("%lu %lu %lu %lu", left.size(), right.size(), left_weights.size(), right_weights.size());
        Matrix<double> matrix(left_weights.size(), right_weights.size());

        std::vector<std::vector<double>> pd_sq;
        std::vector<std::vector<int>> assignment(left_weights.size(), std::vector<int>(right_weights.size(), -1));
        std::vector<int> perm;

        pd_sq = pairwise_distance_sq(left, right, left_weights, right_weights);

        results.ssd.push_back(0);
        // convert pd_sq to a Matrix
        for (int fi = 0; fi < left_weights.size(); ++fi) {
            for (int fj = 0; fj < right_weights.size(); ++fj) {
                matrix(fi, fj) = pd_sq[fi][fj];
                if (fi == fj) results.ssd.back() += pd_sq[fi][fj];
            }
        }

        // Apply Munkres algorithm to matrix.
        Munkres<double> m;
        m.solve(matrix);

        results.matched_ssd.push_back(0);
        for (int row = 0; row < left_weights.size(); row++) {
            for (int col = 0; col < right_weights.size(); col++) {
                assignment[row][col] = matrix(row, col);
                if (matrix(row, col) == 0) results.matched_ssd.back() += pd_sq[row][col];
            }
        }
        ROS_INFO("SSD %f", results.ssd.back());
        results.lifting.push_back(get_permutation(assignment, &results.num_unique));
    }
    return results;
}

match_results clear_matching(std::vector<Phi> const &topic_models,
                             std::function<double(std::vector<int>,
                                                  std::vector<int>,
                                                  double,
                                                  double)> const &similarity_metric = bhattacharyya_coeff<int>) {
    match_results results = {};
    if (topic_models.empty()) return results;

    int const totalNumTopics = std::accumulate(topic_models.begin(),
                                               topic_models.end(),
                                               0,
                                               [](int count, Phi const &next) { return count + next.K; });
    Eigen::MatrixXf P = Eigen::MatrixXf::Constant(totalNumTopics, totalNumTopics, 0);

    size_t i = 0, j_offset = 0;
    for (auto left_idx = 0ul; left_idx < topic_models.size(); ++left_idx) {
        auto const &left = topic_models[left_idx].counts;
        auto const &left_weights = topic_models[left_idx].topic_weights;

        size_t j = j_offset;
        for (auto right_idx = left_idx; right_idx < topic_models.size(); ++right_idx) {
            auto const &right = topic_models[right_idx].counts;
            auto const &right_weights = topic_models[right_idx].topic_weights;

            Eigen::MatrixXf matrix = Eigen::MatrixXf::Constant(left_weights.size(), right_weights.size(), 0);

            results.ssd.push_back(0);
            // convert pd_sq to a Matrix
            for (size_t fi = 0; fi < left_weights.size(); ++fi) {
                for (size_t fj = 0; fj < right_weights.size(); ++fj) {
                    assert(left[fi].size() == right[fj].size());
                    assert(std::accumulate(left[fi].begin(), left[fi].end(), 0) == left_weights[fi]);
                    assert(std::accumulate(right[fj].begin(), right[fj].end(), 0) == right_weights[fj]);
                    matrix(fi, fj) = similarity_metric(left[fi], right[fj], left_weights[fi], right_weights[fj]);
                    double const tolerance = 2e-3;
                    if (!std::isfinite(matrix(fi, fj))) {
                        throw std::logic_error("Invalid entries in similarity matrix!");
                    } else if (matrix(fi, fj) < 0 || matrix(fi, fj) > 1) {
                        if (std::abs(matrix(fi, fj)) <= tolerance) { matrix(fi, fj) = 0.; }
                        else if (std::abs(matrix(fi, fj) - 1.0) <= tolerance) { matrix(fi, fj) = 1.; }
                        else {
                            throw std::logic_error("Invalid entries in similarity matrix!");
                        }
                    }
                    if (left_idx == right_idx && fi == fj && matrix(fi, fj) != 1) {
                        if (std::abs(matrix(fi, fj) - 1.0) <= tolerance) { matrix(fi, fj) = 1.; }
                        else {
                            throw std::logic_error("Invalid entries in similarity matrix!");
                        }
                    }
//                    assert(left_idx != right_idx || fi != fj || matrix(fi, fj) == 1.);
                    if (fi == fj) results.ssd.back() += normed_dist_sq(left[fi], right[fj], left_weights[fi], right_weights[fj]);
                }
            }

            assert(i + left_weights.size() <= P.cols());
            assert(j + right_weights.size() <= P.rows());
            P.block(i, j, left_weights.size(), right_weights.size()) = matrix;
            P.block(j, i, right_weights.size(), left_weights.size()) = matrix.transpose();
            j += right_weights.size();
        }
        j_offset += topic_models[left_idx].K;
        i = j_offset;
    }

//    std::cout << "Pairwise permutation matrix:" << std::endl << P << std::endl;
    assert(P.isApprox((P + P.transpose()) / 2.)); // check symmetric
    assert(P.diagonal() == Eigen::MatrixXf::Identity(totalNumTopics, totalNumTopics).diagonal());
    assert(P.allFinite());

    MultiwayMatcher clearMatcher;
    std::vector<uint32_t> numSmp;
    for (auto const &tm : topic_models) numSmp.push_back(tm.K);
    clearMatcher.initialize(P, numSmp);
    clearMatcher.CLEAR();

    results.num_unique = static_cast<int>(clearMatcher.get_universe_size());
    std::vector<int> assignments = clearMatcher.get_assignments();
    size_t agent_idx = 0, agent_topic_idx = 0;
    results.lifting.emplace_back();
    for (auto const &assignment : assignments) {
        while (agent_topic_idx >= numSmp[agent_idx]) {
            agent_idx++;
            agent_topic_idx = 0;
            assert(agent_idx < numSmp.size());
            results.lifting.emplace_back();
        }
        results.lifting[agent_idx].push_back(assignment);
        agent_topic_idx++;
    }

//    results.matched_ssd.push_back(0);
//    for (int row = 0; row < left_weights.size(); row++) {
//        for (int col = 0; col < right_weights.size(); col++) {
//            assignment[row][col] = matrix(row, col);
//            if (matrix(row, col) == 0) results.matched_ssd.back() += pd_sq[row][col];
//        }
//    }
    // TODO matched ssd calculation -- may need to change Hungarian/ID computation to match
    return results;
}

struct nZWwithPerm {
  std::vector<std::vector<int>> nZW_global;
  std::vector<int> perm;
};

[[deprecated]] nZWwithPerm merge_by_similarity(std::vector<std::vector<int>> nZW_1,
                                               std::vector<int> weightZ_1,
                                               std::vector<std::vector<int>> nZW_2,
                                               std::vector<int> weightZ_2,
                                               std::vector<std::vector<int>> nZW_global,
                                               int K,
                                               int V) {
    // Struct to hold result
    nZWwithPerm nZW_global_with_perm;

    Matrix<double> matrix(K, K);

    std::vector<std::vector<double>> pd_sq;
    std::vector<std::vector<int>> assignment(K, std::vector<int>(K, -1));
    std::vector<std::vector<int>> permuted_nZW_delta_2;
    std::vector<int> perm;

    std::vector<std::vector<int>> prev(nZW_global);

    std::vector<std::vector<int>> nZW_delta_2(K, std::vector<int>(V, 0));

    pd_sq = pairwise_distance_sq(nZW_1, nZW_2, weightZ_1, weightZ_2);

    // convert pd_sq to a Matrix
    for (int fi = 0; fi < K; ++fi) {
        for (int fj = 0; fj < K; ++fj) {
            matrix(fi, fj) = pd_sq[fi][fj];
            // std::cout << pd_sq[fi][fj] << " ";
        }
        // std::cout << std::endl;
    }

    // find optimal permutation of second topic model
    // Display begin matrix state.
    // for ( int row = 0 ; row < K ; row++ ) {
    // 	for ( int col = 0 ; col < K ; col++ ) {
    // 		std::cout.width(2);
    // 		std::cout << matrix(row,col) << ",";
    // 	}
    // 	std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // Apply Munkres algorithm to matrix.
    Munkres<double> m;
    m.solve(matrix);

    // Display solved matrix.
    // for ( int row = 0 ; row < K ; row++ ) {
    // 	for ( int col = 0 ; col < K ; col++ ) {
    // 		std::cout.width(2);
    // 		std::cout << matrix(row,col) << ",";
    // 	}
    // 	std::cout << std::endl;
    // }

    // std::cout << std::endl;

    for (int row = 0; row < K; row++) {
        for (int col = 0; col < K; col++) {
            assignment[row][col] = matrix(row, col);
        }
    }

    // Hungarian hungarian(pd_sq, K, K, HUNGARIAN_MODE_MINIMIZE_COST);
    // hungarian.print_cost();
    // hungarian.solve();
    // assignment = hungarian.assignment();
    // hungarian.print_assignment();
    perm = get_permutation(assignment);

    // We only want to permute the model differences
    for (int k = 0; k < K; ++k) {
        for (int v = 0; v < V; ++v) {
            nZW_delta_2[k][v] = nZW_2[k][v] - prev[k][v];
            if (nZW_delta_2[k][v] < 0) nZW_delta_2[k][v] = 0;
        }
    }
    permuted_nZW_delta_2 = permute(nZW_delta_2, perm);

    for (int k = 0; k < K; ++k) {
        for (int v = 0; v < V; ++v) {
            //for (int r = 0; r < N_robots; ++r) {
            int temp_delta = (nZW_1[k][v] - prev[k][v]);
            if (temp_delta < 0) temp_delta = 0;
            nZW_global[k][v] = nZW_global[k][v] + temp_delta;
            nZW_global[k][v] = nZW_global[k][v] + permuted_nZW_delta_2[k][v];
            //}
        }
    }

    nZW_global_with_perm.nZW_global = nZW_global;
    nZW_global_with_perm.perm = perm;
    return nZW_global_with_perm;
}

#endif //SUNSHINE_PROJECT_ADROST_UTILS_HPP