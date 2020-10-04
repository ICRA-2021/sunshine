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

template<uint32_t pose_dimen = 4, typename GTLabelMap>
SegmentationMatch compute_matches(GTLabelMap const &gt_labels,
                                  size_t const gt_num_topics,
                                  sunshine::Segmentation<std::vector<int>, pose_dimen, int, double> const &topic_seg, bool const warn_missing = true) {
    std::vector<uint32_t> gt_weights(gt_num_topics, 0), topic_weights(topic_seg.observations[0].size(), 0);
    std::vector<std::vector<uint32_t>> matches(topic_seg.observations[0].size(), std::vector<uint32_t>(gt_num_topics, 0));
    double total_weight = 0;
    size_t failed = 0;
    for (auto i = 0; i < topic_seg.observations.size(); ++i) {
        static_assert(pose_dimen == 3 || pose_dimen == 4);
        constexpr size_t offset = (pose_dimen == 4) ? 1 : 0;
        std::array<int, 3> const pose{topic_seg.observation_poses[i][offset], topic_seg.observation_poses[i][1 + offset],
                                      topic_seg.observation_poses[i][2 + offset]};
        auto const topic_label = argmax<>(topic_seg.observations[i]);

        auto iter = gt_labels.find(pose);
        if (iter != gt_labels.end()) {
            auto const &gt_label = iter->second;
            if (gt_label >= gt_num_topics) throw std::invalid_argument("Unexpected gt label -- too high!");
            matches[topic_label][gt_label] += 1;
            gt_weights[gt_label] += 1;
            topic_weights[topic_label] += 1;
            total_weight += 1;
        } else {
            failed += 1;
        }
    }
    if (warn_missing && failed > 0) {
        std::cerr << "Failed to find gt gt_seg for " << failed << " of " << topic_seg.observations.size()
                  << " with topic_seg cell_size = " << topic_seg.cell_size << std::endl;
    }
    return {matches, gt_weights, topic_weights, total_weight};
}

template<typename LabelType, uint32_t pose_dimen = 4, typename GTLabelMap>
SegmentationMatch compute_matches(GTLabelMap const &gt_labels,
                                  size_t const gt_num_topics,
                                  sunshine::Segmentation<LabelType, pose_dimen, int, double> const &topic_seg,
                                  bool const warn_missing = true) {
    auto const num_topics = get_num_topics(topic_seg);
    std::vector<uint32_t> gt_weights( gt_num_topics, 0), topic_weights(num_topics, 0);
    std::vector<std::vector<uint32_t>> matches(num_topics, std::vector<uint32_t>(gt_num_topics, 0));
    double total_weight = 0;
    size_t failed = 0;
    for (auto i = 0; i < topic_seg.observations.size(); ++i) {
        static_assert(pose_dimen == 3 || pose_dimen == 4);
        constexpr size_t offset = (pose_dimen == 4) ? 1 : 0;
        std::array<int, 3> const pose{topic_seg.observation_poses[i][offset], topic_seg.observation_poses[i][1 + offset],
                                      topic_seg.observation_poses[i][2 + offset]};
        uint32_t topic_label;
        if constexpr (std::is_integral_v<LabelType>) topic_label = topic_seg.observations[i];
        else if constexpr (std::is_same_v<LabelType, std::vector<int>>) topic_label = argmax(topic_seg.observations[i]);
        else static_assert(always_false<LabelType>);

        auto iter = gt_labels.find(pose);
        if (iter != gt_labels.end()) {
            auto const &gt_label = iter->second;
            matches[topic_label][gt_label] += 1;
            gt_weights[gt_label] += 1;
            topic_weights[topic_label] += 1;
            total_weight += 1;
        } else {
            failed += 1;
        }
    }
    if (warn_missing && failed > 0) {
        std::cerr << "Failed to find gt gt_seg for " << failed << " of " << topic_seg.observations.size()
                  << " with topic_seg cell_size = " << topic_seg.cell_size << std::endl;
    }
    return {matches, gt_weights, topic_weights, total_weight};
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
    if (nmi > 1) {
        std::cerr << "Invalid NMI value -- NMI: " << nmi << ", MI: " << mi << ", Ex: " << entropy_x << ", Ey: " << entropy_y << std::endl;
    }

    return nmi;
}

template<size_t pose_dimen = 4>
double nmi(sunshine::Segmentation<std::vector<int>, 3, int, double> const &gt_seg,
           sunshine::Segmentation<std::vector<int>, pose_dimen, int, double> const &topic_seg) {
    auto const contingency_table = compute_matches<pose_dimen>(gt_seg.toLookupMap(), get_num_topics(gt_seg), topic_seg);
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
                            uint32_t const total_weight,
                            double const max_entropy = std::numeric_limits<double>::max()) {
//    auto const& n = matches;
    std::vector<uint32_t> const &a = topic_weights;
    std::vector<uint32_t> const &b = gt_weights;
    int64_t const N = total_weight;

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
            auto const max_n = strictNumericCast<uint32_t>(std::min(a[i], b[j]));
            if (max_n == 0) continue;
            auto const min_n = strictNumericCast<uint32_t>(std::max(1l, a[i] + b[j] - N));
            if (min_n > max_n) {
                throw std::logic_error("min_n is greater than max_n!");
            }
            for (auto n_ij = min_n; n_ij <= max_n; ++n_ij) {
                double const lg_n = log_factorial(n_ij);
                double const lg_an = log_factorial(a[i] - n_ij);
                double const lg_bn = log_factorial(b[j] - n_ij);
                double const lg_Nabn = log_factorial((N - a[i] - b[j]) + n_ij);
                int64_t const ai_bj = strictNumericCast<int64_t>(a[i]) * b[j];

                double const first_term = (n_ij * std::log(N * static_cast<double>(n_ij) / ai_bj)) / N;
                double const second_term = std::exp(static_cast<double>(lg_a[i] + lg_b[j] + lg_Na[i] + lg_Nb[j]) - (lg_N + lg_n + lg_an + lg_bn + lg_Nabn));
                emi += first_term * second_term;
            }
        }
    }
    if (emi > max_entropy) {
        throw std::invalid_argument("EMI calculation is invalid based on provided entropy!");
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
    if (ami > 1 + 1e-6) {
        std::cerr << "Invalid AMI value -- AMI: " << ami << ", MI: " << mi << ", EMI: " << emi
                  << ", Ex: " << entropy_x << ", Ey: " << entropy_y << std::endl;
    }

//    double const compare = nmi<pose_dimen>(gt_seg, topic_seg);
//    double const compare_entropy = compute_average_entropy_y(matches, topic_weights, total_weight);
    return std::isnan(ami) ? 0. : ami;
}

template<size_t pose_dimen = 4, bool bound_unit = false>
double ami(sunshine::Segmentation<std::vector<int>, 3, int, double> const &gt_seg,
           sunshine::Segmentation<std::vector<int>, pose_dimen, int, double> const &topic_seg) {
    auto const contingency_table = compute_matches<pose_dimen>(gt_seg.toLookupMap(), get_num_topics(gt_seg), topic_seg);
    auto const &matches = std::get<0>(contingency_table);
    auto const &gt_weights = std::get<1>(contingency_table);
    auto const &topic_weights = std::get<2>(contingency_table);
    double const &total_weight = static_cast<double>(std::get<3>(contingency_table));

    double const mi = compute_mutual_info(matches, gt_weights, topic_weights, total_weight);
    double const ex = entropy<>(gt_weights, total_weight), ey = entropy<>(topic_weights, total_weight);
    double const emi = expected_mutual_info(gt_weights, topic_weights, total_weight, std::max(ex, ey));
    double const ami = compute_ami(contingency_table, mi, emi, ex, ey);
    return (bound_unit) ? std::max(0., std::min(1., ami)) : ami;
}

template<typename GTLabelMap, typename LabelType>
auto compute_metrics(GTLabelMap const &gt_labels,
                     size_t const num_gt_topics,
                     sunshine::Segmentation<LabelType, 4, int, double> const &topic_seg, bool const warn_missing = true) {
    auto const contingency_table = compute_matches<LabelType, 4>(gt_labels, num_gt_topics, topic_seg, warn_missing);
    auto const &matches = std::get<0>(contingency_table);
    auto const &gt_weights = std::get<1>(contingency_table);
    auto const &topic_weights = std::get<2>(contingency_table);
    double const &total_weight = static_cast<double>(std::get<3>(contingency_table));

    double const mi = compute_mutual_info(matches, gt_weights, topic_weights, total_weight);
    double const ex = entropy<>(gt_weights, total_weight), ey = entropy<>(topic_weights, total_weight);
    double const nmi = compute_nmi(contingency_table, mi, ex, ey);
    double const emi = expected_mutual_info(gt_weights, topic_weights, total_weight, std::max(ex, ey));
    double const ami = compute_ami(contingency_table, mi, emi, ex, ey);
    return std::make_tuple(mi, nmi, ami);
};

}

#endif //SUNSHINE_PROJECT_METRIC_HPP
