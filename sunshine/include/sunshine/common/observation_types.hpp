//
// Created by stewart on 3/3/20.
//

#ifndef SUNSHINE_PROJECT_OBSERVATION_TYPES_HPP
#define SUNSHINE_PROJECT_OBSERVATION_TYPES_HPP

#include "opencv2/core.hpp"
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/serialization/array.hpp>
#include "utils.hpp"

namespace sunshine {

struct Observation {
    std::string frame;
    double timestamp;
    uint32_t id;

    Observation(decltype(Observation::frame) frame, decltype(Observation::timestamp) timestamp, decltype(Observation::id) id)
        : frame(std::move(frame))
        , timestamp(timestamp)
        , id(id) { }

    virtual ~Observation() = default;
};

struct ImageObservation : public Observation {
    cv::Mat image;

    ImageObservation(decltype(Observation::frame) frame,
                     decltype(Observation::timestamp) timestamp,
                     decltype(Observation::id) id,
                     cv::Mat image)
        : Observation(std::move(frame), timestamp, id)
        , image(std::move(image)) { }

    ~ImageObservation() override = default;
};

template<typename observation_type, uint32_t pose_dim, typename pose_type = double>
struct SemanticObservation : public Observation {
    typedef observation_type ObservationType;
    typedef pose_type PoseType;
    static uint32_t constexpr PoseDim = pose_dim;

    std::vector<ObservationType> observations;
    std::vector<std::array<PoseType, PoseDim>> observation_poses;

    SemanticObservation(decltype(Observation::frame) const &frame,
                        decltype(Observation::timestamp) timestamp,
                        decltype(Observation::id) id,
                        std::vector<ObservationType> const &observations,
                        std::vector<std::array<PoseType, PoseDim>> const &observationPoses)
        : Observation(frame, timestamp, id)
        , observations(observations)
        , observation_poses(observationPoses) { }

    SemanticObservation(decltype(Observation::frame) const &frame,
                        decltype(Observation::timestamp) timestamp,
                        decltype(Observation::id) id,
                        std::vector<ObservationType> &&observations,
                        std::vector<std::array<PoseType, PoseDim>> &&observationPoses)
        : Observation(frame, timestamp, id)
        , observations(std::move(observations))
        , observation_poses(std::move(observationPoses)) { }

    ~SemanticObservation() override = default;
};

template<typename WordType, uint32_t PoseDim, typename PoseType = double>
struct CategoricalObservation : public SemanticObservation<WordType, PoseDim, PoseType> {
    uint64_t vocabulary_start;
    uint64_t vocabulary_size;

    CategoricalObservation(decltype(Observation::frame) const &frame,
                           decltype(Observation::timestamp) timestamp,
                           decltype(Observation::id) id,
                           std::vector<WordType> const &observations,
                           std::vector<std::array<PoseType, PoseDim>> const &observationPoses,
                           uint64_t vocabularyStart,
                           uint64_t vocabularySize)
        : SemanticObservation<WordType, PoseDim, PoseType>(frame, timestamp, id, observations, observationPoses)
        , vocabulary_start(vocabularyStart)
        , vocabulary_size(vocabularySize) { }

    CategoricalObservation(decltype(Observation::frame) const &frame,
                           decltype(Observation::timestamp) timestamp,
                           decltype(Observation::id) id,
                           std::vector<WordType> &&observations,
                           std::vector<std::array<PoseType, PoseDim>> &&observationPoses,
                           uint64_t vocabularyStart,
                           uint64_t vocabularySize)
        : SemanticObservation<WordType, PoseDim, PoseType>(frame, timestamp, id, std::move(observations), std::move(observationPoses))
        , vocabulary_start(vocabularyStart)
        , vocabulary_size(vocabularySize) { }

    ~CategoricalObservation() override = default;
};

template<typename label_type, uint32_t pose_dim, typename cell_pose_type = int, typename pose_type = double>
struct Segmentation : public SemanticObservation<label_type, pose_dim, cell_pose_type> {
    typedef label_type LabelType;
    typedef pose_type PoseType;
    typedef cell_pose_type CellPoseType;
    static uint32_t constexpr POSE_DIM = pose_dim;

    std::array<PoseType, POSE_DIM> cell_size;

    Segmentation()
        : SemanticObservation<label_type, pose_dim, cell_pose_type>("", 0, 0, {}, {})
        , cell_size({}) { }

    Segmentation(decltype(Observation::frame) const &frame,
                 decltype(Observation::timestamp) timestamp,
                 decltype(Observation::id) id,
                 std::array<PoseType, POSE_DIM> cell_size,
                 std::vector<LabelType> const &labels,
                 std::vector<std::array<CellPoseType, POSE_DIM>> const &poses)
        : SemanticObservation<label_type, pose_dim, cell_pose_type>(frame, timestamp, id, labels, poses)
        , cell_size(cell_size) { }

    Segmentation(decltype(Observation::frame) const &frame,
                 decltype(Observation::timestamp) timestamp,
                 decltype(Observation::id) id,
                 std::array<PoseType, POSE_DIM> cell_size,
                 std::vector<LabelType> &&labels,
                 std::vector<std::array<CellPoseType, POSE_DIM>> &&poses)
        : SemanticObservation<label_type, pose_dim, cell_pose_type>(frame, timestamp, id, labels, poses)
        , cell_size(cell_size) { }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & this->frame;
        ar & this->timestamp;
        ar & this->id;
        ar & this->observations;
        ar & this->observation_poses;
        ar & this->cell_size;
    }

    [[nodiscard]] size_t bytesSize() const {
        auto const headerSize = this->frame.size() + sizeof(this->timestamp) + sizeof(this->id) + sizeof(this->cell_size);
        auto const poseSize = sizeof(this->observation_poses) + sizeof(typename decltype(this->observation_poses)::value_type) * this->observation_poses.capacity();
        auto const dataSize = sizeof(this->observations) + sizeof(typename decltype(this->observations)::value_type) * this->observations.capacity();
        return headerSize + poseSize + dataSize;
    }

    auto toLookupMap() const {
        constexpr size_t offset = (pose_dim == 4) ? 1 : 0;
        if constexpr (is_vector<LabelType>::value) {
            std::unordered_map<std::array<cell_pose_type, 3>, typename LabelType::value_type, hasharray < cell_pose_type, 3>> labels;
            for (auto i = 0; i < this->observations.size(); ++i) {
                typename LabelType::value_type const label = argmax<>(this->observations[i]);
                std::array<int, 3> const pose{this->observation_poses[i][offset], this->observation_poses[i][1 + offset],
                                              this->observation_poses[i][2 + offset]};
                labels.insert({pose, label});
            }
            return labels;
        } else {
            std::unordered_map<std::array<cell_pose_type, 3>, LabelType, hasharray < cell_pose_type, 3>> labels;
            for (auto i = 0; i < this->observations.size(); ++i) {
                LabelType const label = this->observations[i];
                std::array<int, 3> const pose{this->observation_poses[i][offset], this->observation_poses[i][1 + offset],
                                              this->observation_poses[i][2 + offset]};
                labels.insert({pose, label});
            }
            return labels;
        }
    }
};

} // namespace sunshine

#endif // SUNSHINE_PROJECT_OBSERVATION_TYPES_HPP
