//
// Created by stewart on 3/3/20.
//

#ifndef SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
#define SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP

#include <string>
#include <utility>
#include <array>
#include <vector>
#include "opencv2/core.hpp"

namespace sunshine {

struct Observation {
  std::string frame;
  double timestamp;
  uint32_t id;

  Observation(decltype(Observation::frame) frame, decltype(Observation::timestamp) timestamp, decltype(Observation::id) id)
        : frame(std::move(frame))
        , timestamp(timestamp)
        , id(id) {}

  virtual ~Observation() = default;
};

struct ImageObservation : public Observation {
  cv::Mat image;

  ImageObservation(decltype(Observation::frame) frame,
                   decltype(Observation::timestamp) timestamp,
                   decltype(Observation::id) id,
                   cv::Mat image)
        : Observation(std::move(frame), timestamp, id)
        , image(std::move(image)) {}

  ~ImageObservation() override = default;
};

template<typename _ObservationType, uint32_t _PoseDim, typename _PoseType = double>
struct SemanticObservation : public Observation {
  typedef _ObservationType ObservationType;
  typedef _PoseType PoseType;
  static uint32_t constexpr PoseDim = _PoseDim;

  std::vector<ObservationType> observations;
  std::vector<std::array<PoseType, PoseDim>> observation_poses;

  SemanticObservation(decltype(Observation::frame) const &frame,
                      decltype(Observation::timestamp) timestamp,
                      decltype(Observation::id) id,
                      std::vector<ObservationType> const &observations,
                      std::vector<std::array<PoseType, PoseDim>> const &observationPoses)
        : Observation(frame, timestamp, id)
        , observations(observations)
        , observation_poses(observationPoses) {}

  SemanticObservation(decltype(Observation::frame) const &frame,
                      decltype(Observation::timestamp) timestamp,
                      decltype(Observation::id) id,
                      std::vector<ObservationType> &&observations,
                      std::vector<std::array<PoseType, PoseDim>> &&observationPoses)
        : Observation(frame, timestamp, id)
        , observations(std::move(observations))
        , observation_poses(std::move(observationPoses)) {}

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
        , vocabulary_size(vocabularySize) {}

  CategoricalObservation(decltype(Observation::frame) const &frame,
                         decltype(Observation::timestamp) timestamp,
                         decltype(Observation::id) id,
                         std::vector<WordType>&& observations,
                         std::vector<std::array<PoseType, PoseDim>>&& observationPoses,
                         uint64_t vocabularyStart,
                         uint64_t vocabularySize)
        : SemanticObservation<WordType, PoseDim, PoseType>(frame, timestamp, id, std::move(observations), std::move(observationPoses))
        , vocabulary_start(vocabularyStart)
        , vocabulary_size(vocabularySize) {}

  ~CategoricalObservation() override = default;
};

}

#endif //SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
