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

  Observation(std::string frame, long timestamp, int32_t id)
        : frame(std::move(frame))
        , timestamp(timestamp)
        , id(id) {}

  virtual ~Observation() = default;
};

struct ImageObservation : public Observation {
  cv::Mat image;

  ImageObservation(std::string frame, long timestamp, int32_t id, cv::Mat image)
        : Observation(std::move(frame), timestamp, id)
        , image(std::move(image)) {}

  ~ImageObservation() override = default;
};

template<typename ObservationType, uint32_t PoseDim, typename PoseType = double>
struct SemanticObservation : public Observation {
  std::vector<ObservationType> observations;
  std::vector<std::array<PoseType, PoseDim>> observation_poses;

  SemanticObservation(std::string const &frame,
                      long timestamp,
                      int32_t id,
                      std::vector<ObservationType> const &observations,
                      std::vector<std::array<PoseType, PoseDim>> const &observationPoses)
        : Observation(frame, timestamp, id)
        , observations(observations)
        , observation_poses(observationPoses) {}

  ~SemanticObservation() override = default;
};

template<typename WordType, uint32_t PoseDim, typename PoseType = double>
struct CategoricalObservation : public SemanticObservation<WordType, PoseDim, PoseType> {
  uint64_t vocabulary_start;
  uint64_t vocabulary_size;

  CategoricalObservation(std::string const &frame,
                         long timestamp,
                         int32_t id,
                         std::vector<WordType> const &observations,
                         std::vector<std::array<PoseType, PoseDim>> const &observationPoses,
                         uint64_t vocabularyStart,
                         uint64_t vocabularySize)
        : SemanticObservation<WordType, PoseDim, PoseType>(frame, timestamp, id, observations, observationPoses)
        , vocabulary_start(vocabularyStart)
        , vocabulary_size(vocabularySize) {}

  ~CategoricalObservation() override = default;
};

}

#endif //SUNSHINE_PROJECT_SUNSHINE_TYPES_HPP
