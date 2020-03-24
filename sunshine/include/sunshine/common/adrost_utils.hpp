#ifndef SUNSHINE_PROJECT_ADROST_UTILS_HPP
#define SUNSHINE_PROJECT_ADROST_UTILS_HPP

#include <cmath>
#include <munkres/munkres.h>
#include <clear/MultiwayMatcher.hpp>
#include <Eigen/Core>
#include <numeric>
#include <ostream>
#include <istream>
#include <utility>
#include <ros/console.h> // TODO remove? only needed for logging
#include "sunshine_types.hpp"
// #include "adapters/boostmatrixadapter.h"

namespace sunshine {

template<typename L, typename R = L> using SimilarityMetric = std::function<double(std::vector<L>, std::vector<R>, double, double)>;
template<typename L, typename R = L> using DistanceMetric = std::function<double(std::vector<L>, std::vector<R>, double, double)>;

double unit_round(double value, double epsilon = 2e-3) {
    if (value >= 0 && value <= 1) return value;
    if (std::abs(value) <= epsilon) return 0;
    if (std::abs(value - 1) <= epsilon) return 1;
    throw std::logic_error("Value is not within [0,1]");
}

struct match_scores {
  public:
    int K = 0; // number of clusters
    std::vector<size_t> cluster_sizes = {}; // number of elements in each cluster (Kx1)
    std::vector<double> mscd = {}; // mean of squared cluster distances; closer to 0 is better
    std::vector<double> silhouette = {}; // closer to +1 is better
    std::vector<double> davies_bouldin = {}; // closer to 0 is better // TODO: implement
//    std::vector<double> calinski_harabasz = {}; // TODO: decide whether to use (this metric isn't great because it is unbounded above)

    match_scores(std::vector<Phi> const &topic_models,
                 std::vector<std::vector<int>> const &lifting,
                 DistanceMetric<double> const &metric = nullptr) {
        assert(topic_models.size() == lifting.size());

        for (auto agent = 0ul; agent < topic_models.size(); ++agent) {
            auto const &data = topic_models[agent].counts;
            assert(data.size() == lifting[agent].size());
            for (auto topic = 0ul; topic < data.size(); ++topic) {
                auto const cluster = lifting[agent][topic];
                auto const scale = topic_models[agent].topic_weights[topic];
                if (cluster >= K) {
                    K = cluster + 1;
                    cluster_sizes.resize(K, 0);
                    cluster_centers.resize(K, std::vector<double>(topic_models[agent].V, 0));
                    cluster_scales.resize(K, 0.);
                }

                sorted_points.insert(clusterEnd(sorted_points, cluster), std::vector<double>{data[topic].begin(), data[topic].end()});
                point_scales.insert(clusterEnd(point_scales, cluster), scale);

                cluster_sizes[cluster]++;
                std::transform(cluster_centers[cluster].begin(),
                               cluster_centers[cluster].end(),
                               data[topic].begin(),
                               cluster_centers[cluster].begin(),
                               [scale](double const& sum, int const& val){return sum + (static_cast<double>(val) / ((scale > 0) ? scale : 1.));});
                cluster_scales[cluster] += (scale > 0) ? 1 : 0;
            }
        }

        if (metric != nullptr) compute_scores(metric);
    }

    void compute_scores(DistanceMetric<double> const &metric) {
        std::vector<std::vector<double>> dispersion_matrix = {}; // group dispersion matrix (NxN)
        this->mscd.resize(0);
        this->mscd.reserve(K);
        this->silhouette.resize(0);
        this->silhouette.reserve(K);
        this->davies_bouldin.resize(0);
        this->davies_bouldin.reserve(K);

        dispersion_matrix.resize(sorted_points.size(), std::vector<double>(sorted_points.size(), 0.0));
        for (auto i = 0ul; i < sorted_points.size(); ++i) {
            for (auto j = i; j < sorted_points.size(); ++j) {
                dispersion_matrix[i][j] = metric(sorted_points[i], sorted_points[j], point_scales[i], point_scales[j]);
                dispersion_matrix[j][i] = dispersion_matrix[i][j];
            }
        }

        size_t cluster_start = 0, cluster_end = 0;
        for (auto k = 0; k < K; k++, cluster_start = cluster_end) {
            cluster_end += cluster_sizes[k];
            double mscd_k = 0;
            double silhouette_k = 0;
            assert(std::abs(std::accumulate(cluster_centers[k].begin(), cluster_centers[k].end(), 0.0) - cluster_scales[k]) <= 1e-3);
            if (!cluster_sizes[k]) ROS_WARN("Empty cluster detected when computing metrics!");
            if (cluster_sizes[k] > 1) {
                for (auto i = cluster_start; i < cluster_end; ++i) {
                    mscd_k += std::pow(metric(sorted_points[i], cluster_centers[k], point_scales[i], cluster_scales[k]), 2);

                    double silhouette_a = 0;
                    double silhouette_b = std::numeric_limits<double>::max() / 2., silhouette_b_tmp = 0;
                    int tmp_k = 0;
                    int tmp_start = 0;
                    for (auto j = 0ul; j < sorted_points.size(); j++) {
                        while (j - tmp_start >= cluster_sizes[tmp_k]) {
                            if (tmp_k != k && cluster_sizes[tmp_k] > 0) {
                                silhouette_b = std::min(silhouette_b, silhouette_b_tmp / cluster_sizes[tmp_k]);
                            }
                            tmp_start += cluster_sizes[tmp_k];
                            tmp_k++;
                            silhouette_b_tmp = 0;
                            assert(tmp_k < K);
                        }

                        if (tmp_k == k) { silhouette_a += dispersion_matrix[i][j]; }
                        else { silhouette_b_tmp += dispersion_matrix[i][j]; }
                    }
                    if (tmp_k != k && cluster_sizes[tmp_k] > 0) {
                        silhouette_b = std::min(silhouette_b, silhouette_b_tmp / cluster_sizes[tmp_k]);
                    }
                    silhouette_a /= std::max(cluster_sizes[k] - 1, 1ul);
                    silhouette_k += (std::max(silhouette_a, silhouette_b) != 0)
                                    ? (silhouette_b - silhouette_a) / std::max(silhouette_a, silhouette_b)
                                    : 0.;
                }
            } else { silhouette_k = 0.0; }

            mscd.push_back(mscd_k / std::max(cluster_sizes[k], 1ul));
            silhouette.push_back(silhouette_k / std::max(cluster_sizes[k], 1ul));
        }

        for (auto k = 0; k < K; k++) {
            double db_k = 0;
            for (auto k2 = 0; k2 < K; k2++) {
                if (k == k2) continue;
                auto const R = (mscd[k] + mscd[k2]) / metric(cluster_centers[k], cluster_centers[k2], cluster_scales[k], cluster_scales[k2]);
                db_k = std::max(db_k, R);
            }
            davies_bouldin.push_back(db_k);
        }
    }

  private:
    std::vector<std::vector<double>> sorted_points = {}; // coordinates of points, sorted by clusters (NxV)
    std::vector<double> point_scales = {}; // scales of points (i.e. sum of elements in each point)
    std::vector<std::vector<double>> cluster_centers = {}; // coordinates of cluster centers (KxV)
    std::vector<double> cluster_scales = {}; // scales of cluster centers (i.e. sum of elements in each cluster center)

    template<typename T>
    typename T::const_iterator clusterEnd(T const &t, int cluster) {
        assert(cluster < K);
        auto iter = t.cbegin();
        for (size_t idx = 0; idx <= cluster; ++idx) {
            assert(cluster_sizes[idx] <= std::distance(iter, t.cend()));
            iter += cluster_sizes[idx];
        }
        return iter;
    }
};

struct match_results {
  int num_unique = -1;
  std::vector<std::vector<int>> lifting = {};
  std::vector<double> ssd = {};
};

/**
 * Computes the squared euclidean distance between two vectors, v and w
 * @param v the first vector<float>
 * @param w the second vector<float>
 * @returns float - the squared eucl. distance between v and w
**/
template<typename T>
double normed_dist_sq(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    if (v.size() != w.size()) throw std::invalid_argument("Vector sizes do not match");
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
double gaussian_kernel_similarity(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    auto const dist = normed_dist_sq(v, w, scale_v, scale_w);
    auto const length_scale = 1.;
    return std::exp(-dist / length_scale);
}

template<typename T>
double l1_distance(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    if (v.size() != w.size()) throw std::invalid_argument("Vector sizes do not match");
    double distance = 0.0;
    double diff;

    if (scale_v == 0 && scale_w == 0) { return 0.; }
    else if (scale_v == 0 || scale_w == 0) { return 1.; }

    double const invscale_v = 1. / scale_v, invscale_w = 1. / scale_w;

    for (auto i = 0ul; i < v.size(); ++i) {
        // diff = 10000.0f*v[i] - 10000.0f*w[i];
        diff = (v[i] * invscale_v) - (w[i] * invscale_w);
        distance += std::abs(diff);
    }

    return distance / 2.;
}

template<typename T>
double l1_similarity(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    return 1 - l1_distance(v, w, scale_v, scale_w);
}

template<typename T>
double l2_distance(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    return std::sqrt(normed_dist_sq(v, w, scale_v, scale_w) / 2.);
}

template<typename T>
double l2_similarity(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    return 1 - l2_distance(v, w, scale_v, scale_w);
}

template<typename T>
double kl_div(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    if (v.size() != w.size()) throw std::invalid_argument("Vector sizes do not match");
    if (scale_v == 0 && scale_w == 0) { return 0.; }
    else if (scale_v == 0 || scale_w == 0) { return std::numeric_limits<double>::infinity(); }

    double divergence = 0;
    for (auto i = 0ul; i < v.size(); ++i) {
        divergence += (v[i])
                      ? v[i] * std::log2((v[i] * scale_w) / (w[i] * scale_v))
                      : 0.;
    }
    return divergence / scale_v;
}

template<typename T>
double symmetric_kl_div(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    return kl_div(v, w, scale_v, scale_w) + kl_div(w, v, scale_w, scale_v);
}

template<typename T>
double jensen_shannon_div(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    if (scale_v == 0 && scale_w == 0) { return 0.; }
    else if (scale_v == 0 || scale_w == 0) { return 1; }

    assert (v.size() == w.size());
    std::vector<double> m;
    std::transform(v.begin(),
                   v.end(),
                   w.begin(),
                   std::back_inserter(m),
                   [=](T const &left, T const &right) { return left / scale_v + right / scale_w; });
    double const scale_m = std::accumulate(m.begin(), m.end(), 0.0);
    assert(std::abs(scale_m - 2.) < 1e-3);
    // TODO validate this code
    auto const js_div = (kl_div<double>(std::vector<double>(v.begin(), v.end()), m, scale_v, scale_m)
          + kl_div<double>(std::vector<double>(w.begin(), w.end()), m, scale_w, scale_m)) / 2.;
    return unit_round(js_div);
}

template<typename T>
double jensen_shannon_dist(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    auto const div = jensen_shannon_div(v, w, scale_v, scale_w);
    return unit_round(std::sqrt(div));
}

template<typename T>
double jensen_shannon_similarity(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    return 1. - jensen_shannon_dist(v, w, scale_v, scale_w);
}

template<typename T>
double jensen_shannon_similarity_v2(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    return 1. - jensen_shannon_div(v, w, scale_v, scale_w);
}

/**
 * Computes the cosine similarity two vectors, v and w
 * @param v the first vector<float>
 * @param w the second vector<float>
**/
template<typename T>
double cosine_similarity(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    if (v.size() != w.size()) throw std::invalid_argument("Vector sizes do not match");
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
    assert(norm_v >= 0 && norm_w >= 0 && dot_p >= 0);
    return unit_round(dot_p / (norm_v * norm_w));
}

template<typename T>
double angular_distance(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    auto const angle = std::acos(cosine_similarity(v, w, scale_v, scale_w));
    return unit_round(angle / M_PI);
}

/**
 * Computes the Bhattacharyya coefficient between two vectors, v and w
 * @param v the first vector<float>
 * @param w the second vector<float>
**/
template<typename T>
double bhattacharyya_coeff(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    if (v.size() != w.size()) throw std::invalid_argument("Vector sizes do not match");
    if (scale_v == 0 && scale_w == 0) { return 1.; }
    else if (scale_v == 0 || scale_w == 0) { return 0.; }

    double const rt_scale = std::sqrt(scale_v * scale_w);
    double sum = 0.0;
    for (auto i = 0ul; i < v.size(); ++i) {
        if (v[i] > 0 && w[i] > 0) {
            sum += std::sqrt(static_cast<double>(v[i]) * w[i]) / rt_scale;
        }
    }
    return unit_round(sum);
}

template<typename T>
double hellinger_dist(std::vector<T> const &v, std::vector<T> const &w, double scale_v = 1., double scale_w = 1.) {
    return std::sqrt(1. - bhattacharyya_coeff(v, w, scale_v, scale_w));
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
template<typename T, typename S = T>
std::vector<std::vector<double>> compute_all_pairs(std::vector<std::vector<T>> const &a,
                                                   std::vector<std::vector<T>> const &b,
                                                   std::vector<S> const &scale_a,
                                                   std::vector<S> const &scale_b,
                                                   DistanceMetric<T> const &metric) {
    std::vector<std::vector<double>> pd(a.size(), std::vector<double>(b.size(), 0.0));
    for (auto i = 0ul; i < a.size(); ++i) {
        for (auto j = 0ul; j < b.size(); ++j) {
            pd[i][j] = metric(a[i],
                              b[j],
                              (scale_a.empty())
                              ? 1
                              : scale_a[i],
                              (scale_b.empty())
                              ? 1.
                              : scale_b[j]);
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

    for (auto i = 1ul; i < topic_models.size(); ++i) {
        auto const &right = topic_models[i].counts;
        auto const &right_weights = topic_models[i].topic_weights;
        Matrix<double> matrix(left_weights.size(), right_weights.size());

        std::vector<std::vector<double>> pd_sq = compute_all_pairs<int>(left, right, left_weights, right_weights, normed_dist_sq<int>);
        std::vector<std::vector<int>> assignment(left_weights.size(), std::vector<int>(right_weights.size(), -1));

        results.ssd.push_back(0);
        for (uint64_t row = 0; row < left_weights.size(); row++) {
            assignment[row][row] = 0;
            results.ssd.back() += pd_sq[row][row];
        }
        results.lifting.push_back(get_permutation(assignment, &results.num_unique));
    }
    return results;
}

match_results sequential_hungarian_matching(std::vector<Phi> const &topic_models, DistanceMetric<int> const &metric = normed_dist_sq<int>) {
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

    for (auto i = 1ul; i < topic_models.size(); ++i) {
        auto const &right = topic_models[i].counts;
        auto const &right_weights = topic_models[i].topic_weights;
//        ROS_WARN("%lu %lu %lu %lu", left.size(), right.size(), left_weights.size(), right_weights.size());
        Matrix<double> matrix(left_weights.size(), right_weights.size());

        std::vector<std::vector<int>> assignment(left_weights.size(), std::vector<int>(right_weights.size(), -1));
        std::vector<int> perm;

        auto const pd_sq = compute_all_pairs(left, right, left_weights, right_weights, metric);

        results.ssd.push_back(0);
        // convert pd_sq to a Matrix
        for (int fi = 0; fi < left_weights.size(); ++fi) {
            for (int fj = 0; fj < right_weights.size(); ++fj) {
                matrix(fi, fj) = pd_sq[fi][fj];
                if (fi == fj) results.ssd.back() += normed_dist_sq(left[fi], right[fj], left_weights[fi], right_weights[fj]);
            }
        }

        // Apply Munkres algorithm to matrix.
        Munkres<double> m;
        m.solve(matrix);

        for (int row = 0; row < left_weights.size(); row++) {
            for (int col = 0; col < right_weights.size(); col++) {
                assignment[row][col] = matrix(row, col);
            }
        }

//        ROS_INFO("SSD %f", results.ssd.back());
        results.lifting.push_back(get_permutation(assignment, &results.num_unique));
    }
    return results;
}

match_results clear_matching(std::vector<Phi> const &topic_models,
                             SimilarityMetric<int> const &similarity_metric = bhattacharyya_coeff<int>,
                             bool enforce_distinctness = false) {
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
                        ROS_ERROR("Invalid entry %f in similarity matrix!", matrix(fi, fj));
                        throw std::logic_error("Invalid entries in similarity matrix!");
                    } else { matrix(fi, fj) = unit_round(matrix(fi, fj), tolerance); }
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

//    std::cout << "Pairwise similarity matrix:" << std::endl;
//    Eigen::IOFormat CleanFmt(3, 0, ",", "\n", "[", "]");
//    std::cout << P.format(CleanFmt) << std::endl;

    MultiwayMatcher clearMatcher(enforce_distinctness);
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

match_results match_topics(std::string const &method, std::vector<Phi> const &topic_models) {
    if (method == "id") {
        return id_matching(topic_models);
    } else if (method == "hungarian-l2") {
        return sequential_hungarian_matching(topic_models, l2_distance<int>);
    } else if (method == "hungarian-l1") {
        return sequential_hungarian_matching(topic_models, l1_distance<int>);
    } else if (method == "hungarian-js") {
        return sequential_hungarian_matching(topic_models, jensen_shannon_dist<int>);
    } else if (method == "hungarian-js2") {
        return sequential_hungarian_matching(topic_models, jensen_shannon_div<int>);
    } else if (method == "hungarian-angle") {
        return sequential_hungarian_matching(topic_models, angular_distance<int>);
    } else if (method == "hungarian-hg") {
        return sequential_hungarian_matching(topic_models, hellinger_dist<int>);
    } else if (method == "clear-l1") {
        return clear_matching(topic_models, l1_similarity<int>, false);
    } else if (method == "clear-l2") {
        return clear_matching(topic_models, l2_similarity<int>, false);
    } else if (method == "clear-gk") {
        return clear_matching(topic_models, gaussian_kernel_similarity<int>, false);
    } else if (method == "clear-bh") {
        return clear_matching(topic_models, bhattacharyya_coeff<int>, false);
    } else if (method == "clear-cosine") {
        return clear_matching(topic_models, cosine_similarity<int>, false);
    } else if (method == "clear-js") {
        return clear_matching(topic_models, jensen_shannon_similarity<int>, false);
    } else if (method == "clear-js2") {
        return clear_matching(topic_models, jensen_shannon_similarity_v2<int>, false);
    } else if (method == "clear-distinct-l1") {
        return clear_matching(topic_models, l1_similarity<int>, true);
    } else if (method == "clear-distinct-l2") {
        return clear_matching(topic_models, l2_similarity<int>, true);
    } else if (method == "clear-distinct-gk") {
        return clear_matching(topic_models, gaussian_kernel_similarity<int>, true);
    } else if (method == "clear-distinct-bh") {
        return clear_matching(topic_models, bhattacharyya_coeff<int>, true);
    } else if (method == "clear-distinct-cosine") {
        return clear_matching(topic_models, cosine_similarity<int>, true);
    } else if (method == "clear-distinct-js") {
        return clear_matching(topic_models, jensen_shannon_similarity<int>, true);
    } else if (method == "clear-distinct-js2") {
        return clear_matching(topic_models, jensen_shannon_similarity_v2<int>, true);
    } else {
        throw std::logic_error(method + " is not recognized.");
    }
}
}

#endif //SUNSHINE_PROJECT_ADROST_UTILS_HPP