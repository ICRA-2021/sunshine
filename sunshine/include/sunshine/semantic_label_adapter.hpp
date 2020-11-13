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

class SemanticLabelAdapter : public Adapter<SemanticLabelAdapter, ImageObservation, CategoricalObservation<int, 2, int>>
{
    typedef CategoricalObservation<int, 2, int> Output;
    UniqueStore<std::array<uint8_t, 3>, int> unique_obs;
    int const num_labels;
    int const step_size;
    double seq_start = 0.0;
    double const seq_duration;

  public:

    std::unique_ptr<Output> operator()(ImageObservation const *imgObs) {
        std::vector<int> observations;
        std::vector<std::array<int, 2>> observation_poses;
        observation_poses.reserve(imgObs->image.rows * imgObs->image.cols);
        int const half_step = step_size / 2;
        for (int y = 0; y < imgObs->image.rows; y += step_size) {
            for (int x = 0; x < imgObs->image.cols; x += step_size) {
                std::vector<int> counts(num_labels, 0);
                for (int dx = 0; (dx < step_size) && (x + dx < imgObs->image.cols); ++dx) {
                    for (int dy = 0; (dy < step_size) && (y + dy < imgObs->image.rows); ++dy) {
                        auto const &rgb = imgObs->image.at<cv::Vec3b>(cv::Point(x + dx, y + dy));
                        int const label = unique_obs.get_id({rgb[0], rgb[1], rgb[2]});
                        if (label > num_labels) throw std::runtime_error("SemanticLabelAdapter found too many unique labels.");
                        counts[label]++;
                    }
                }
                observations.push_back(argmax<>(counts));
                observation_poses.push_back({x + half_step, y +  half_step});
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
            : Adapter<SemanticLabelAdapter, ImageObservation, CategoricalObservation<int, 2, int>>(),
              num_labels(nh->template param<int>("num_labels", nh->template param<int>("K", 10))),
              seq_duration(nh->template param<double>("seq_duration", 0)),
              step_size(nh->template param<int>("step_size", 8)) {
        assert(step_size > 0);
    }
};
}

#endif //SUNSHINE_PROJECT_SEMANTIC_LABEL_ADAPTER_HPP
