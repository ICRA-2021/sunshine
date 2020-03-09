//
// Created by stewart on 3/4/20.
//

#ifndef SUNSHINE_PROJECT_ROS_CONVERSIONS_HPP
#define SUNSHINE_PROJECT_ROS_CONVERSIONS_HPP

#include "sunshine/common/sunshine_types.hpp"
#include <sunshine_msgs/WordObservation.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <tf2/convert.h>

namespace sunshine {

template<typename WordObservation>
sunshine_msgs::WordObservation toRosMsg(WordObservation const &in, geometry_msgs::TransformStamped observationTransform = {}) {
    static_assert(std::is_integral<typename WordObservation::ObservationType>::value, "Only integral word types are supported!");
    static_assert(sizeof(typename WordObservation::ObservationType) <= sizeof(decltype(sunshine_msgs::WordObservation::words)::value_type),
                  "WordType is too large!");
    auto constexpr poseDim = WordObservation::PoseDim;

    sunshine_msgs::WordObservation out{};
    out.header.frame_id = in.frame;
//    out.header.seq = in.id; // deprecate?
    out.header.stamp = ros::Time(in.timestamp);
    out.seq = in.id;
    out.source = ""; // unused
    out.vocabulary_begin = in.vocabulary_start;
    out.vocabulary_size = in.vocabulary_size;
    out.observation_transform = observationTransform;

    out.words.reserve(in.observations.size());
    out.word_pose.reserve(in.observations.size() * poseDim);
    out.word_scale = {}; // uninitialized

    for (auto i = 0ul, j = 0ul; i < in.observations.size(); ++i) {
        out.words.emplace_back(in.observations[i]);
        assert(in.observation_poses[i].size() == poseDim);
        for (auto d = 0u; d < poseDim; ++d) {
            out.word_pose.emplace_back(static_cast<double>(in.observation_poses[i][d]));
        }
    }
    return out;
}

template<typename WordType, uint32_t PoseDim, typename WordPoseType>
CategoricalObservation <WordType, PoseDim, WordPoseType> fromRosMsg(sunshine_msgs::WordObservation const &in) {
    static_assert(PoseDim <= 4, "Only up to 4-d poses are currently supported for ROS conversion");
    static_assert(std::is_integral<WordType>::value, "Only integral word types are supported!");
    static_assert(sizeof(WordType) <= sizeof(decltype(sunshine_msgs::WordObservation::words)::value_type), "WordType is too large!");

    auto out = CategoricalObservation<WordType, PoseDim, WordPoseType>((in.observation_transform.child_frame_id.empty())
                                                                       ? in.header.frame_id
                                                                       : in.observation_transform.child_frame_id,
                                                                       in.header.stamp.toSec(),
                                                                       in.seq,
                                                                       {},
                                                                       {},
                                                                       in.vocabulary_begin,
                                                                       in.vocabulary_size);

    out.observations.reserve(in.words.size());
    out.observation_poses.reserve(in.words.size());

    uint32_t const inputDim = (in.words.empty())
                              ? 0
                              : in.word_pose.size() / in.words.size();
    if (inputDim != PoseDim) throw std::logic_error("PoseDim does not match pose dimension of input message!");
    if (in.words.size() * inputDim != in.word_pose.size()) { throw std::invalid_argument("Malformed sunshine_msgs::WordObservation"); }

    for (auto i = 0ul, j = 0ul; i < in.words.size(); ++i, j += inputDim) {
        out.observations.emplace_back(in.words[i]);
        assert(j + PoseDim <= in.word_pose.size());

        geometry_msgs::Point point;
        switch (inputDim) {
            case 3:
                point.z = in.word_pose[j + 2];
            case 2:
                point.y = in.word_pose[j + 1];
            case 1:
                point.x = in.word_pose[j];
                break;
            default:
                throw std::logic_error("Cannot handle more than 3 input dimensions");
        }
        std::array<WordPoseType, PoseDim> wordPose = {0};
        tf2::doTransform(point, point, in.observation_transform);
        switch (inputDim) {
            case 3:
                wordPose[2] = point.z;
            case 2:
                wordPose[1] = point.y;
            case 1:
                wordPose[0] = point.x;
                break;
            default:
                throw std::logic_error("Cannot handle more than 3 input dimensions");
        }
        out.observation_poses.push_back(wordPose);
    }
    return out;
}

}

#endif //SUNSHINE_PROJECT_ROS_CONVERSIONS_HPP
