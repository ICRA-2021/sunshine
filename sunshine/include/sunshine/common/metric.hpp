//
// Created by stewart on 2020-08-30.
//

#ifndef SUNSHINE_PROJECT_METRIC_HPP
#define SUNSHINE_PROJECT_METRIC_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

#include "utils.hpp"
#include "observation_types.hpp"

namespace sunshine {

typedef std::tuple<std::vector<std::vector<uint32_t>>, std::vector<uint32_t>, std::vector<uint32_t>, uint32_t> SegmentationMatch;

template<typename Container>
double entropy(Container const &container, double weight = 1.0) {
    double sum = 0;
    for (auto const &val : container) {
        if (val > 0) {
            double const pv = (weight == 1.0) ? val : (val / weight);
            sum += pv * std::log(pv);
        } else {
            assert(val == 0);
        }
    }
    return -sum;
}

template<uint32_t pose_dimen = 4>
SegmentationMatch compute_matches(sunshine::Segmentation<std::vector<int>, 3, int, double> const &gt_seg,
                                  sunshine::Segmentation<std::vector<int>, pose_dimen, int, double> const &topic_seg) {
    std::map<std::array<int, 3>, uint32_t> gt_labels, topic_labels;
    std::vector<uint32_t> gt_weights(gt_seg.observations[0].size(), 0), topic_weights(topic_seg.observations[0].size(), 0);
    std::vector<std::vector<uint32_t>> matches(topic_seg.observations[0].size(), std::vector<uint32_t>(gt_seg.observations[0].size(), 0));
    double total_weight = 0;
    for (auto i = 0; i < gt_seg.observations.size(); ++i) {
        auto const label = argmax<>(gt_seg.observations[i]);
        gt_labels.insert({gt_seg.observation_poses[i], label});
    }
    for (auto i = 0; i < topic_seg.observations.size(); ++i) {
        static_assert(pose_dimen == 3 || pose_dimen == 4);
        constexpr size_t offset = (pose_dimen == 4) ? 1 : 0;
        std::array<int, 3> const pose{topic_seg.observation_poses[i][offset], topic_seg.observation_poses[i][1 + offset],
                                      topic_seg.observation_poses[i][2 + offset]};
        auto const topic_label = argmax<>(topic_seg.observations[i]);
        topic_labels.insert({pose, topic_label});

        auto iter = gt_labels.find(pose);
        if (iter != gt_labels.end()) {
            auto const &gt_label = iter->second;
            matches[topic_label][gt_label] += 1;
            gt_weights[gt_label] += 1;
            topic_weights[topic_label] += 1;
            total_weight += 1;
        } else {
            std::cerr << "Failed to find gt gt_seg for pose!" << std::endl;
        }
    }
    return {matches, gt_weights, topic_weights, total_weight};
}

template<typename SegmentationType>
auto get_num_topics(SegmentationType const &seg) {
    if constexpr (is_vector<typename decltype(seg.observations)::value_type>::value) {
        return seg.observations[0].size();
    } else if constexpr (std::is_integral_v<typename decltype(seg.observations)::value_type>) {
        return *std::max_element(seg.observations.begin(), seg.observations.end()) + 1;
    } else {
        static_assert(always_false < SegmentationType > );
    }
}

template<uint32_t pose_dimen = 4>
SegmentationMatch compute_matches(sunshine::Segmentation<int, 3, int, double> const &gt_seg,
                                  sunshine::Segmentation<int, pose_dimen, int, double> const &topic_seg) {
    std::map<std::array<int, 3>, uint32_t> gt_labels, topic_labels;
    auto const gt_num_topics = get_num_topics(gt_seg);
    auto const num_topics = get_num_topics(topic_seg);
    std::vector<uint32_t> gt_weights( gt_num_topics, 0), topic_weights(num_topics, 0);
    std::vector<std::vector<uint32_t>> matches(num_topics, std::vector<uint32_t>(gt_num_topics, 0));
    double total_weight = 0;
    for (auto i = 0; i < gt_seg.observations.size(); ++i) {
        auto const label = gt_seg.observations[i];
        gt_labels.insert({gt_seg.observation_poses[i], label});
    }
    for (auto i = 0; i < topic_seg.observations.size(); ++i) {
        static_assert(pose_dimen == 3 || pose_dimen == 4);
        constexpr size_t offset = (pose_dimen == 4) ? 1 : 0;
        std::array<int, 3> const pose{topic_seg.observation_poses[i][offset], topic_seg.observation_poses[i][1 + offset],
                                      topic_seg.observation_poses[i][2 + offset]};
        auto const topic_label = topic_seg.observations[i];
        topic_labels.insert({pose, topic_label});

        auto iter = gt_labels.find(pose);
        if (iter != gt_labels.end()) {
            auto const &gt_label = iter->second;
            matches[topic_label][gt_label] += 1;
            gt_weights[gt_label] += 1;
            topic_weights[topic_label] += 1;
            total_weight += 1;
        } else {
            std::cerr << "Failed to find gt gt_seg for pose!" << std::endl;
        }
    }
    return {matches, gt_weights, topic_weights, total_weight};
}

template<typename LabelType, uint32_t pose_dimen = 4, typename CountType = double>
std::pair<std::vector<std::vector<CountType>>, CountType> compute_cooccurences(sunshine::Segmentation<LabelType, 3, int, double> const &gt_seg,
                                                                               sunshine::Segmentation<LabelType, pose_dimen, int, double> const &topic_seg) {
    auto const N = get_num_topics(topic_seg);
    auto const M = get_num_topics(gt_seg);
    CountType total_weight = 0;
    std::vector<std::vector<CountType>> cooccurences(N, std::vector<CountType>(M, 0));
    std::map<std::array<int, 3>, LabelType> gt_labels;
    for (auto obs = 0; obs < gt_seg.observations.

            size();

         ++obs) {
        gt_labels.insert({gt_seg.observation_poses[obs], gt_seg.observations[obs]});
    }
    for (auto obs = 0ul; obs < topic_seg.observation_poses.

            size();

         ++obs) {
        static_assert(pose_dimen == 3 || pose_dimen == 4);
        constexpr size_t offset = (pose_dimen == 4) ? 1 : 0;
        std::array<int, 3> const pose{topic_seg.observation_poses[obs][offset], topic_seg.observation_poses[obs][1 + offset],
                                      topic_seg.observation_poses[obs][2 + offset]};
        auto iter = gt_labels.find(pose);
        if (iter != gt_labels.

                                     end()

                ) {
            auto const &observed = topic_seg.observations[obs];
            auto const &actual = iter->second;
            if constexpr (is_vector<LabelType>::value) {
                double const weight_observed = std::accumulate(observed.begin(), observed.end(), 0.0);
                double const weight_actual = std::accumulate(actual.begin(), actual.end(), 0.0);
                double prod_weight = weight_observed * weight_actual;
                for (auto i = 0ul; i < N; ++i) {
                    for (auto j = 0ul; j < M; ++j) {
                        cooccurences[i][j] += static_cast
                                                      <CountType>(observed[i] * actual[j]) / prod_weight;
                    }
                }
            } else {
                cooccurences[observed][actual] += 1;
            }
            total_weight += 1;
        } else {
            std::cerr << "Failed to find gt gt_seg for pose!" << std::endl;
        }
    }
    return {cooccurences, total_weight};
}

double compute_mutual_info(std::vector<std::vector<uint32_t >> const &matches,
                           std::vector<uint32_t> const &gt_weights,
                           std::vector<uint32_t> const &topic_weights,
                           double const total_weight) {
    std::vector<double> px, py;
    px.reserve(topic_weights.size());
    py.reserve(gt_weights.size());
    for (auto const &topic_weight
            : topic_weights) {
        px.push_back(topic_weight / total_weight);
    }
    for (auto const &gt_weight
            : gt_weights) {
        py.push_back(gt_weight / total_weight);
    }

    double mi = 0;
    double sum_pxy = 0;
    for (auto i = 0; i < matches.size(); ++i) {
        for (auto j = 0; j < matches[i].size(); ++j) {
            auto const pxy = matches[i][j] / total_weight;
            if (pxy > 0) {
                mi += pxy * log(pxy / (px[i] * py[j]));
            }
            sum_pxy += pxy;
        }
    }
    assert(sum_pxy >= 0.99 && sum_pxy <= 1.01);

    return mi;
}

double compute_average_entropy_y(std::vector<std::vector<uint32_t >> const &matches,
                                 std::vector<uint32_t> const &topic_weights,
                                 double const total_weight) {
    std::vector<double> px;
    px.reserve(topic_weights.size());
    for (auto const &topic_weight
            : topic_weights) {
        px.push_back(topic_weight / total_weight);
    }

    double exp_ey = 0;
    for (auto i = 0; i < matches.size(); ++i) {
        exp_ey += entropy<>(matches[i], topic_weights[i]) * px[i];
    }
    return exp_ey;
}

double compute_nmi(SegmentationMatch const &contingency_table, double const mi, double const entropy_x, double const entropy_y) {
    auto const &matches = std::get<0>(contingency_table);
    auto const &gt_weights = std::get<1>(contingency_table);
    auto const &topic_weights = std::get<2>(contingency_table);
    double const &total_weight = static_cast<double>(std::get<3>(contingency_table));

    if ((entropy_x <= 0. ^ entropy_y <= 0.) && mi != 0) {
        std::cerr << "Somehow entropy_x = " << entropy_x << " <= 0 or entropy_y = " << entropy_y << " <= 0 but MI " << mi << " > 0"
                  << std::endl;
    }
    double const nmi = (entropy_x <= 0 && entropy_y <= 0) ? 1. : ((mi == 0) ? 0. : (mi / sqrt(entropy_x * entropy_y)));

    return nmi;
}

template<size_t pose_dimen = 4>
double nmi(sunshine::Segmentation<std::vector<int>, 3, int, double> const &gt_seg,
           sunshine::Segmentation<std::vector<int>, pose_dimen, int, double> const &topic_seg) {
    auto const contingency_table = compute_matches<pose_dimen>(gt_seg, topic_seg);
    auto const &matches = std::get<0>(contingency_table);
    auto const &gt_weights = std::get<1>(contingency_table);
    auto const &topic_weights = std::get<2>(contingency_table);
    double const &total_weight = static_cast<double>(std::get<3>(contingency_table));
    double const mi = compute_mutual_info(matches, gt_weights, topic_weights, total_weight);
    double const ex = entropy<>(gt_weights, total_weight), ey = entropy<>(topic_weights, total_weight);
    double const nmi = compute_nmi(contingency_table, mi, ex, ey);
    return nmi;
}

double expected_mutual_info(std::vector<uint32_t> const &gt_weights,
                            std::vector<uint32_t> const &topic_weights,
                            uint32_t const total_weight) {
//    auto const& n = matches;
    auto const &a = topic_weights;
    auto const &b = gt_weights;
    auto const &N = total_weight;

    auto const log_factorial = [](uint32_t val) { return std::lgamma(val + 1); };

    std::vector<double> lg_b, lg_a, lg_Nb, lg_Na;
    double const lg_N = log_factorial(N);
    lg_a.reserve(topic_weights.size());
    lg_Na.reserve(topic_weights.size());
    lg_b.reserve(gt_weights.size());
    lg_Nb.reserve(gt_weights.size());
    for (auto const &topic_weight
            : topic_weights) {
        lg_a.push_back(log_factorial(topic_weight));
        lg_Na.push_back(log_factorial(N - topic_weight));
    }
    for (auto const &gt_weight
            : gt_weights) {
        lg_b.push_back(log_factorial(gt_weight));
        lg_Nb.push_back(log_factorial(N - gt_weight));
    }

    double emi = 0;
    for (auto i = 0; i < topic_weights.size(); ++i) {
        for (auto j = 0; j < gt_weights.size(); ++j) {
            uint32_t const min_n = std::max(1u, a[i] + b[j] - N);
            uint32_t const max_n = std::min(a[i], b[j]);
            for (auto n_ij = min_n; n_ij <= max_n; ++n_ij) {
                double const lg_n = log_factorial(n_ij);
                double const lg_an = log_factorial(a[i] - n_ij);
                double const lg_bn = log_factorial(b[j] - n_ij);
                double const lg_Nabn = log_factorial(N - a[i] - b[j] + n_ij);

                double const first_term = static_cast<double>(n_ij) * std::log(N * static_cast<double>(n_ij) / (a[i] * b[j])) / N;
                double const second_term = std::exp((lg_a[i] + lg_b[j] + lg_Na[i] + lg_Nb[j]) - (lg_N + lg_n + lg_an + lg_bn + lg_Nabn));
                emi += first_term * second_term;
            }
        }
    }
    return emi;
}

double compute_ami(SegmentationMatch const &contingency_table,
                   double const mi,
                   double const emi,
                   double const entropy_x,
                   double const entropy_y) {
    auto const &matches = std::get<0>(contingency_table);
    auto const &gt_weights = std::get<1>(contingency_table);
    auto const &topic_weights = std::get<2>(contingency_table);
    double const &total_weight = static_cast<double>(std::get<3>(contingency_table));

    double const ami = (entropy_x <= 0. && entropy_y <= 0.) ? 1 : ((mi - emi) / (std::max(entropy_x, entropy_y) - emi));

//    double const compare = nmi<pose_dimen>(gt_seg, topic_seg);
//    double const compare_entropy = compute_average_entropy_y(matches, topic_weights, total_weight);
    return std::isnan(ami) ? 0. : ami;
}

template<size_t pose_dimen = 4>
double ami(sunshine::Segmentation<std::vector<int>, 3, int, double> const &gt_seg,
           sunshine::Segmentation<std::vector<int>, pose_dimen, int, double> const &topic_seg) {
    auto const contingency_table = compute_matches<pose_dimen>(gt_seg, topic_seg);
    auto const &matches = std::get<0>(contingency_table);
    auto const &gt_weights = std::get<1>(contingency_table);
    auto const &topic_weights = std::get<2>(contingency_table);
    double const &total_weight = static_cast<double>(std::get<3>(contingency_table));

    double const mi = compute_mutual_info(matches, gt_weights, topic_weights, total_weight);
    double const emi = expected_mutual_info(gt_weights, topic_weights, total_weight);
    double const ex = entropy<>(gt_weights, total_weight), ey = entropy<>(topic_weights, total_weight);
    double const ami = compute_ami(contingency_table, mi, emi, ex, ey);
    return ami;
}
}

#endif //SUNSHINE_PROJECT_METRIC_HPP
