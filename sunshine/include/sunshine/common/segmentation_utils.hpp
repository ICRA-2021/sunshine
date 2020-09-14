//
// Created by stewart on 2020-09-10.
//

#ifndef SUNSHINE_PROJECT_SEGMENTATION_UTILS_HPP
#define SUNSHINE_PROJECT_SEGMENTATION_UTILS_HPP

#include "utils.hpp"
#include "observation_types.hpp"
#include "matching_utils.hpp"

namespace sunshine {


template<typename LabelType, uint32_t pose_dimen = 4, typename CountType = double>
std::pair<std::vector<std::vector<CountType>>, CountType> compute_cooccurences(sunshine::Segmentation<LabelType, 3, int, double> const &gt_seg,
                                                                               sunshine::Segmentation<LabelType, pose_dimen, int, double> const &topic_seg) {
    auto const N = get_num_topics(topic_seg);
    auto const M = get_num_topics(gt_seg);
    CountType total_weight = 0;
    std::vector<std::vector<CountType>> cooccurences(N, std::vector<CountType>(M, 0));
    std::map<std::array<int, 3>, LabelType> gt_labels;
    assert(gt_seg.observation_poses.size() == gt_seg.observations.size());
    for (auto obs = 0; obs < gt_seg.observations.size(); ++obs) {
        gt_labels.insert({gt_seg.observation_poses[obs], gt_seg.observations[obs]});
    }
    static_assert(std::tuple_size_v<typename decltype(topic_seg.observation_poses)::value_type> == pose_dimen);
    assert(topic_seg.observation_poses.size() == topic_seg.observations.size());
    for (auto obs = 0ul; obs < topic_seg.observation_poses.size(); ++obs) {
        static_assert(pose_dimen == 3 || pose_dimen == 4);
        constexpr size_t offset = (pose_dimen == 4) ? 1 : 0;
        std::array<int, 3> const pose{topic_seg.observation_poses[obs][offset], topic_seg.observation_poses[obs][1 + offset],
                                      topic_seg.observation_poses[obs][2 + offset]};
        auto iter = gt_labels.find(pose);
        if (iter != gt_labels.end()) {
            auto const &observed = topic_seg.observations[obs];
            auto const &actual = iter->second;
            if constexpr (is_vector<LabelType>::value) {
                double const weight_observed = std::accumulate(observed.begin(), observed.end(), 0.0);
                double const weight_actual = std::accumulate(actual.begin(), actual.end(), 0.0);
                double prod_weight = weight_observed * weight_actual;
                for (auto i = 0ul; i < N; ++i) {
                    for (auto j = 0ul; j < M; ++j) {
                        assert(i < observed.size() && j < actual.size());
                        cooccurences[i][j] += static_cast<CountType>(observed[i] * actual[j]) / prod_weight;
                    }
                }
            } else {
                assert(observed < cooccurences.size() && actual < cooccurences[observed].size());
                cooccurences[observed][actual] += 1;
            }
            total_weight += 1;
        } else {
            std::cerr << "Failed to find gt gt_seg for pose " << pose
                      << " with gt_seg cell_size = " << gt_seg.cell_size
                      << " and topic_seg cell_size = " << topic_seg.cell_size << std::endl;
        }
    }
    return {cooccurences, total_weight};
}

template<uint32_t PoseDim = 4, typename Container>
std::unique_ptr<Segmentation<int, PoseDim, int, double>> merge(Container const &segmentations,
                                                               std::vector<std::vector<int>> const &lifting = {}) {
    auto merged = std::make_unique<Segmentation<int, PoseDim, int, double>>(segmentations[0]->frame,
                                                                            segmentations[0]->timestamp,
                                                                            segmentations[0]->id,
                                                                            segmentations[0]->cell_size,
                                                                            std::vector<int>(),
                                                                            segmentations[0]->observation_poses);
    for (auto i = 1; i < segmentations.size(); ++i) {
        merged->observation_poses.insert(merged->observation_poses.end(),
                                         segmentations[i]->observation_poses.begin(),
                                         segmentations[i]->observation_poses.end());
    }
    if (lifting.empty()) {
        for (auto const &map : segmentations) {
            if constexpr (std::is_integral_v<typename decltype(map->observations)::value_type>) {
                merged->observations.insert(merged->observations.end(), map->observations.begin(), map->observations.end());
            } else {
                std::transform(map->observations.begin(),
                               map->observations.end(),
                               std::back_inserter(merged->observations),
                               argmax<std::vector<int>>);
            }
        }
    } else {
        for (auto i = 0; i < segmentations.size(); ++i) {
            if constexpr (std::is_integral_v<typename decltype(segmentations[i]->observations)::value_type>) {
                std::transform(segmentations[i]->observations.begin(),
                               segmentations[i]->observations.end(),
                               std::back_inserter(merged->observations),
                               [i, &lifting](int const &obs) {
                                   assert(obs < lifting[i].size());
                                   return lifting[i][obs]; });
            } else {
                std::transform(segmentations[i]->observations.begin(),
                               segmentations[i]->observations.end(),
                               std::back_inserter(merged->observations),
                               [i, &lifting](std::vector<int> const &obs) {
                                   assert(argmax(obs) < lifting[i].size());
                                   return lifting[i][argmax(obs)]; });
            }
        }
    }
    return merged;
}

template<typename LabelType, uint32_t LeftPoseDim = 3, uint32_t RightPoseDim = 3>
void align(Segmentation<LabelType, LeftPoseDim, int, double> &segmentation, Segmentation<LabelType, RightPoseDim, int, double> const &reference) {
    auto const cooccurrence_data = compute_cooccurences(reference, segmentation);
    auto const &counts = cooccurrence_data.first;
    std::vector<std::vector<double>> costs(counts[0].size(), std::vector<double>(counts.size(), 0.0));
    for (auto i = 0ul; i < counts.size(); ++i) {
        for (auto j = 0ul; j < counts[0].size(); ++j) {
            costs[j][i] = std::accumulate(counts[i].begin(), counts[i].end(), 0.0) - 2 * counts[i][j];
        }
    }
    int num_topics = counts.size();
    auto const lifting = get_permutation(hungarian_assignments(costs), &num_topics);
    for (auto &obs : segmentation.observations) {
        if constexpr (std::is_integral_v<LabelType>) obs = lifting[obs];
        else {
            std::remove_reference_t<decltype(obs)> copy{obs};
            assert(&copy != &obs);
            assert(obs.size() <= lifting.size());
            for (auto i = 0ul; i < obs.size(); ++i) obs[i] = copy[lifting[i]];
        }
    }
}
}

#endif //SUNSHINE_PROJECT_SEGMENTATION_UTILS_HPP
