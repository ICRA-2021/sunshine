//
// Created by stewart on 3/4/20.
//

#ifndef SUNSHINE_PROJECT_ROS_CONVERSIONS_HPP
#define SUNSHINE_PROJECT_ROS_CONVERSIONS_HPP

#include <ros/console.h>
#include "sunshine/common/observation_types.hpp"
#include "sunshine/common/sunshine_types.hpp"
#include "sunshine/common/utils.hpp"
#include <sunshine_msgs/WordObservation.h>
#include <sunshine_msgs/TopicModel.h>
#include <sunshine_msgs/TopicMap.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2/convert.h>

#include <utility>

namespace sunshine {

template<typename WordType, uint32_t PoseDim, typename PoseType = double>
sunshine_msgs::WordObservation toRosMsg(CategoricalObservation<WordType, PoseDim, PoseType> const &in, geometry_msgs::TransformStamped observationTransform) {
    static_assert(std::is_integral<WordType>::value, "Only integral word types are supported!");
    static_assert(sizeof(WordType) <= sizeof(decltype(sunshine_msgs::WordObservation::words)::value_type),
                  "WordType is too large!");
    auto constexpr poseDim = PoseDim;

    sunshine_msgs::WordObservation out{};
    out.header.frame_id = in.frame;
//    out.header.seq = in.id; // deprecate?
    out.header.stamp = ros::Time(in.timestamp);
    out.seq = in.id;
    out.source = ""; // unused
    out.vocabulary_begin = in.vocabulary_start;
    out.vocabulary_size = in.vocabulary_size;
    out.observation_transform = std::move(observationTransform);

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

template<typename WordType, uint32_t PoseDim, typename PoseType = double>
sunshine_msgs::WordObservation toRosMsg(CategoricalObservation<WordType, PoseDim, PoseType> const &in) {
    static_assert(std::is_integral<WordType>::value, "Only integral word types are supported!");
    static_assert(sizeof(WordType) <= sizeof(decltype(sunshine_msgs::WordObservation::words)::value_type),
                  "WordType is too large!");
    auto constexpr poseDim = PoseDim;

    sunshine_msgs::WordObservation out{};
    out.header.frame_id = in.frame;
    //    out.header.seq = in.id; // deprecate?
    out.header.stamp = ros::Time(in.timestamp);
    out.seq = in.id;
    out.source = ""; // unused
    out.vocabulary_begin = in.vocabulary_start;
    out.vocabulary_size = in.vocabulary_size;
    out.observation_transform = {};
    out.observation_transform.transform.rotation.w = 1;

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

ImageObservation fromRosMsg(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    return ImageObservation(msg->header.frame_id, msg->header.stamp.toSec(), msg->header.seq, std::move(img_ptr->image));
}

Phi fromRosMsg(sunshine_msgs::TopicModel const &topic_model) {
    Phi out(topic_model.identifier, topic_model.K, topic_model.V, {}, topic_model.topic_weights);
    assert(*std::min_element(topic_model.topic_weights.cbegin(), topic_model.topic_weights.cend()) >= 0);
    out.counts.reserve(topic_model.K);
    for (auto i = 0ul; i < topic_model.K; ++i) {
        out.counts
           .emplace_back(topic_model.phi.begin() + i * topic_model.V,
                         (i + 1 < topic_model.K)
                         ? topic_model.phi.begin() + (i + 1) * topic_model.V
                         : topic_model.phi.end());
        assert(out.counts[i].size() == topic_model.V);
    }
    return out;
}

sunshine_msgs::TopicModel toRosMsg(Phi const &phi) {
    sunshine_msgs::TopicModel topicModel;
    topicModel.K = phi.K;
    topicModel.V = phi.V;
    topicModel.identifier = phi.id;
    topicModel.topic_weights = phi.topic_weights;
    topicModel.phi.reserve(phi.K * phi.V);
    for (auto i = 0ul; i < phi.K; ++i) {
        auto topicDist = (std::vector<int>) phi.counts[i];
        topicModel.phi.insert(topicModel.phi.end(), topicDist.begin(), topicDist.end());
    }
    assert(topicModel.phi.size() == phi.K * phi.V);
    return topicModel;
}

template<typename LabelType, uint32_t PoseDim>
sunshine_msgs::TopicMap toRosMsg(Segmentation<LabelType, PoseDim, int, double> const& segmentation) {
    sunshine_msgs::TopicMap map;
    map.header.frame_id = segmentation.frame;
    map.header.stamp = map.header.stamp.fromSec(segmentation.timestamp);
    map.seq = segmentation.id;
    static_assert(PoseDim == 3 || PoseDim == 4, "Unsupported pose dimensionality");
    map.cell_time = (PoseDim == 4) ? segmentation.cell_size[0] : -1;
    constexpr uint32_t offset = (PoseDim == 4);
    map.cell_width = {segmentation.cell_size[offset], segmentation.cell_size[offset + 1], segmentation.cell_size[offset + 2]};
    map.cell_poses.reserve(segmentation.observation_poses.size() * 3);
    auto const cell_size = segmentation.cell_size;
    std::for_each(segmentation.observation_poses.begin(), segmentation.observation_poses.end(), [&map, offset, &cell_size](std::array<int, PoseDim> const& pose){
        std::array<double, 3> world_pose = {};
        if constexpr (PoseDim == 3) {
            world_pose = toWordPose<3, int, double>(pose, cell_size);
        } else if constexpr(PoseDim == 4) {
            world_pose = toWordPose<3, int, double>({pose[1], pose[2], pose[3]}, {cell_size[1], cell_size[2], cell_size[3]});
        } else {
            static_assert(always_false<PoseDim>);
        }
        map.cell_poses.insert(map.cell_poses.end(), world_pose.begin(), world_pose.end());
    });
    if constexpr (std::is_same_v<LabelType, int>) {
        map.cell_topics = segmentation.observations;
    } else if constexpr (std::is_same_v<LabelType, std::vector<int>>) {
        std::transform(segmentation.observations.begin(), segmentation.observations.end(), std::back_inserter(map.cell_topics), argmax<std::vector<int>>);
    } else {
        static_assert(always_false<LabelType>);
    }
    return map;
}

}

#endif //SUNSHINE_PROJECT_ROS_CONVERSIONS_HPP
