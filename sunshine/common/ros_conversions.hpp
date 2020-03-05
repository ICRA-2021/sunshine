//
// Created by stewart on 3/4/20.
//

#ifndef SUNSHINE_PROJECT_ROS_CONVERSIONS_HPP
#define SUNSHINE_PROJECT_ROS_CONVERSIONS_HPP

#include "sunshine_types.hpp"
#include <sunshine_msgs/WordObservation.h>

namespace sunshine {

template<typename WordType, uint32_t PoseDim, typename WordPoseType>
sunshine_msgs::WordObservation toRosMsg(CategoricalObservation<WordType, PoseDim, WordPoseType> const& in) {
    static_assert(std::is_integral<WordType>::value, "Only integral word types are supported!");
    static_assert(sizeof(WordType) <= sizeof(decltype(sunshine_msgs::WordObservation::words)::value_type), "WordType is too large!");

    sunshine_msgs::WordObservation out{};
    out.header.frame_id = in.frame;
    out.header.seq = in.id; // deprecate?
    out.header.stamp = ros::Time(in.timestamp);
    out.seq = in.id;
    out.source = in.source;
    out.vocabulary_start = in.vocabulary_start;
    out.vocabulary_size = in.vocabulary_size;
    out.observation_transform = {}; // uninitialized

    out.words.reserve(in.observations.size());
    out.word_pose.reserve(in.observations.size() * PoseDim);
    out.word_scale = {}; // uninitialized

    for (auto i = 0ul, j = 0ul; i < in.observations.size(); ++i, j+=PoseDim) {
        out.words.emplace_back(in.observations[i]);
        assert(j + PoseDim <= in.observation_poses.size());
        for (auto d = 0u; d < PoseDim; ++d) {
            out.word_pose.emplace_back(in.observation_poses[j + d]);
        }
    }
    return out;
}

}

#endif //SUNSHINE_PROJECT_ROS_CONVERSIONS_HPP
