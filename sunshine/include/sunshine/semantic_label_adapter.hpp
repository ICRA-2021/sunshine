//
// Created by stewart on 2020-07-13.
//

#ifndef SUNSHINE_PROJECT_SEMANTIC_LABEL_ADAPTER_HPP
#define SUNSHINE_PROJECT_SEMANTIC_LABEL_ADAPTER_HPP

#include "common/observation_adapters.hpp"
#include "common/utils.hpp"
#include "common/object_stores.hpp"
#include "rost_adapter.hpp"

namespace sunshine {

template<typename LabelType = int>
class SemanticLabelAdapter : public Adapter<SemanticLabelAdapter<LabelType>, ImageObservation, CategoricalObservation<LabelType, 2, int>>
{
    typedef CategoricalObservation<LabelType, 2, int> Output;
    UniqueStore<std::array<uint8_t, 3>, LabelType> unique_obs;
    LabelType const num_labels;
    double seq_start = 0.0;
    double const seq_duration;

  public:

    std::unique_ptr<Output> operator()(ImageObservation const *imgObs) {
        std::vector<LabelType> observations;
        std::vector<std::array<int, 2>> observation_poses;
        observation_poses.reserve(imgObs->image.rows * imgObs->image.cols);
        for (int y = 0; y < imgObs->image.rows; ++y) {
            for (int x = 0; x < imgObs->image.cols; ++x) {
                auto const &rgb = imgObs->image.at<cv::Vec3b>(cv::Point(x, y));
                LabelType const label = unique_obs.get_id({rgb[0], rgb[1], rgb[2]});
                if (label > num_labels) throw std::runtime_error("SemanticLabelAdapter found too many unique labels.");
                observations.push_back(label);
                observation_poses.push_back({x, y});
            }
        }

        if (seq_start == 0) {
            seq_start = imgObs->timestamp;
        }
        uint32_t id = (seq_duration == 0)
                      ? imgObs->id
                      : static_cast<uint32_t>((imgObs->timestamp - seq_start) / seq_duration);
        uint32_t const &vocabulary_start = 0;
        uint32_t const &vocabulary_size = num_labels;
        std::string const &frame = imgObs->frame;
        double const &timestamp = imgObs->timestamp;

        return std::make_unique<Output>(frame, timestamp, id, observations, observation_poses, vocabulary_start, vocabulary_size);
    }

    template<typename ParameterServer>
    explicit SemanticLabelAdapter(ParameterServer *nh)
            : Adapter<SemanticLabelAdapter, ImageObservation, CategoricalObservation<LabelType, 2, int>>(),
              num_labels(nh->template param<int>("num_labels", 10)),
              seq_duration(nh->template param<double>("seq_duration", 0)) {
    }
};
}

#endif //SUNSHINE_PROJECT_SEMANTIC_LABEL_ADAPTER_HPP
