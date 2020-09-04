//
// Created by stewart on 2020-09-04.
//

#ifndef SUNSHINE_PROJECT_2D_ADAPTER_HPP
#define SUNSHINE_PROJECT_2D_ADAPTER_HPP

#include "common/observation_adapters.hpp"
#include "common/observation_types.hpp"
#include <cstdint>

namespace sunshine {

template <size_t pose_dim = 3>
class Word2DAdapter : public Adapter<Word2DAdapter<pose_dim>, CategoricalObservation<int, 2, int>, CategoricalObservation<int, pose_dim, double>> {
public:
    std::unique_ptr<CategoricalObservation<int, pose_dim, double>> operator()(CategoricalObservation<int, 2, int> const * wordObs2d) const {
        auto const num_words = wordObs2d->observations.size();

        std::vector<std::array<double, 3>> observation_pose;
        observation_pose.reserve(num_words);
        for (size_t i = 0; i < num_words; ++i) {
            double const u = wordObs2d->observation_poses[i][0], v = wordObs2d->observation_poses[i][1];
            observation_pose.push_back({u, v, 0.0});
        }

        return std::make_unique<CategoricalObservation<int, 3, double>>("map",
            wordObs2d->timestamp,
            wordObs2d->id,
            wordObs2d->observations,
            std::move(observation_pose),
                    wordObs2d->vocabulary_start,
            wordObs2d->vocabulary_size);
    }
    using Adapter<Word2DAdapter<pose_dim>, CategoricalObservation<int, 2, int>, CategoricalObservation<int, pose_dim, double>>::operator();
};

template <size_t pose_dim = 3>
class Image2DAdapter : public Adapter<Image2DAdapter<pose_dim>, ImageObservation, SemanticObservation<std::array<uint8_t, 3>, pose_dim, double>> {
public:
    std::unique_ptr<SemanticObservation<std::array<uint8_t, 3>, 3, double>> operator()(ImageObservation const* wordObs2d) const {
        auto const num_words = wordObs2d->image.rows * wordObs2d->image.cols;

        std::vector<std::array<uint8_t, 3>> observations;
        std::vector<std::array<double, 3>> observation_pose;
        observations.reserve(num_words);
        observation_pose.reserve(num_words);
        for (int y = 0; y < wordObs2d->image.rows; ++y) {
            for (int x = 0; x < wordObs2d->image.cols; ++x) {
                auto const& rgb = wordObs2d->image.at<cv::Vec3b>(cv::Point(x, y));
                observations.push_back({rgb[0], rgb[1], rgb[2]});
                observation_pose.push_back({static_cast<double>(x), static_cast<double>(y), 0.0});
            }
        }

        return std::make_unique<SemanticObservation<std::array<uint8_t, 3>, 3, double>>("map",
        wordObs2d->timestamp,
        wordObs2d->id,
        std::move(observations),
                std::move(observation_pose));
    }
    using Adapter<Image2DAdapter<pose_dim>, ImageObservation, SemanticObservation<std::array<uint8_t, 3>, pose_dim, double>>::operator();
};


}

#endif //SUNSHINE_PROJECT_2D_ADAPTER_HPP
