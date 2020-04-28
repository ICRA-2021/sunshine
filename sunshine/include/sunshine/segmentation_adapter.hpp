//
// Created by stewart on 3/25/20.
//

#ifndef SUNSHINE_PROJECT_SEGMENTATION_ADAPTER_HPP
#define SUNSHINE_PROJECT_SEGMENTATION_ADAPTER_HPP

#include "common/observation_adapters.hpp"
#include "common/utils.hpp"
#include "common/object_stores.hpp"
#include "rost_adapter.hpp"

namespace sunshine {


template<typename T>
struct is_vector : public std::false_type {
};

template<typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type {
};

template<typename Input, typename LabelType, uint32_t PoseDim, typename CellPoseType = int, typename PoseType = int>
class SegmentationAdapter : public Adapter<SegmentationAdapter<Input, LabelType, PoseDim, CellPoseType, PoseType>, Input, Segmentation<LabelType, PoseDim, CellPoseType, PoseType>> {
  protected:
    static size_t constexpr POSEDIM = PoseDim;
    std::array<PoseType, POSEDIM> const cell_size;

    template<typename ParameterServer>
    static inline decltype(cell_size) computeCellSize(ParameterServer *nh) {
        double const cell_size_space = nh->template param<double>("cell_space", 1);
        std::string const cell_size_string = nh->template param<std::string>("cell_size", "");
        if (!cell_size_string.empty()) {
            return readNumbers<POSEDIM, 'x', PoseType>(cell_size_string);
        } else if constexpr (POSEDIM == 2) {
            return {static_cast<PoseType>(cell_size_space), static_cast<PoseType>(cell_size_space)};
        } else if constexpr (POSEDIM == 3) {
            return {static_cast<PoseType>(cell_size_space), static_cast<PoseType>(cell_size_space), static_cast<PoseType>(cell_size_space)};
        }
        return {};
    }

  public:
    template<typename ParameterServer>
    explicit SegmentationAdapter(ParameterServer *paramServer)
    : cell_size(computeCellSize(paramServer)) {}
};

template<typename LabelType, typename CellPoseType = int, typename PoseType = int>
class ImageSegmentationAdapter : public SegmentationAdapter<ImageObservation, LabelType, 2, CellPoseType, PoseType> {
    static size_t constexpr POSEDIM = 2;
    UniqueStore<std::array<uint8_t, 3>> unique_colors;

    std::array<uint8_t, 3> fromVec(cv::Vec3b const& vec) {
        return {vec[0], vec[1], vec[2]};
    }

  public:
    template<typename ParameterServer>
    explicit ImageSegmentationAdapter(ParameterServer *paramServer)
          : SegmentationAdapter<ImageObservation, LabelType, POSEDIM, CellPoseType, PoseType>(paramServer) {}

    std::unique_ptr<Segmentation<LabelType, POSEDIM, CellPoseType, PoseType>> operator()(std::unique_ptr<ImageObservation> const &imageObs) {
        typedef Segmentation<LabelType, POSEDIM, CellPoseType, PoseType> Output;
        auto segmentation = std::make_unique<Output>(imageObs->frame, imageObs->timestamp, imageObs->id, this->cell_size, {}, {});
        if (imageObs->image.rows == 0 || imageObs->image.cols == 0) return std::move(segmentation);

        size_t const numCells = (std::floor(imageObs->image.cols / this->cell_size[0]) + 1) * (std::floor(imageObs->image.rows / this->cell_size[1]) + 1);
        segmentation.observation_poses.reserve(numCells);
        segmentation.observations.reserve(numCells);

        std::map<std::array<CellPoseType, POSEDIM>, std::map<size_t, size_t>> counts;
        for (auto y = 0; y < imageObs->image.rows; y++) {
            for (auto x = 0; x < imageObs->image.cols; x++) {
                std::array<CellPoseType, POSEDIM> const pose = ROSTAdapter<2, int, int>::toCellId({x, y}, this->cell_size);
                auto const tmp = imageObs->image.at<cv::Vec3b>(cv::Point(x, y));
                size_t const id = unique_colors.get_id(fromVec(imageObs->image.at<cv::Vec3b>(cv::Point(x, y))));
                if (auto iter = counts.find(pose); iter != counts.end()) {
                    if (auto countIter = iter->second.find(id); countIter != iter->second.end()) {
                        countIter->second++;
                    } else {
                        iter->second.emplace(id, 1);
                    }
                } else {
                    counts.emplace(pose, std::map<size_t, size_t>{});
                    iter = counts.find(pose);
                    if (iter == counts.end()) throw std::logic_error("Failed to insert new pose");
                    iter->second.emplace(id, 1);
                }
            }
        }

        if constexpr (std::is_integral_v<LabelType>) {
            // TODO Find max count
        } else if constexpr (is_vector<LabelType>::value) {
            static_assert(std::is_integral_v<typename LabelType::value_type>);
            for (auto const& entry : counts) {
                segmentation->observation_poses.push_back(entry.first);
                LabelType labelVec;
                labelVec.resize(unique_colors.size(), 0);
                for (auto const& idCount : entry.second) {
                    labelVec[idCount.first] = idCount.second;
                }
                segmentation->observations.push_back(labelVec);
            }
        } else {
            static_assert(always_false<LabelType>);
        }

        return std::move(segmentation);
    }
};

template<typename ObservationType, typename LabelType, uint32_t PoseDim = 3, typename CellPoseType = int, typename PoseType = double>
class SemanticSegmentationAdapter : public SegmentationAdapter<SemanticObservation<ObservationType, PoseDim, PoseType>, LabelType, PoseDim, CellPoseType, PoseType> {
    static size_t constexpr POSEDIM = PoseDim;
    UniqueStore<ObservationType> unique_obs;

  public:
    template<typename ParameterServer>
    explicit SemanticSegmentationAdapter(ParameterServer *paramServer)
          : SegmentationAdapter<SemanticObservation<ObservationType, PoseDim, PoseType>, LabelType, PoseDim, CellPoseType, PoseType>(paramServer) {}

    std::unique_ptr<Segmentation<LabelType, POSEDIM, CellPoseType, PoseType>> operator()(std::unique_ptr<SemanticObservation<ObservationType, PoseDim, PoseType>> const &obs) {
        typedef Segmentation<LabelType, POSEDIM, CellPoseType, PoseType> Output;
        auto segmentation = std::make_unique<Output>(obs->frame, obs->timestamp, obs->id, this->cell_size, std::vector<LabelType>(), std::vector<std::array<CellPoseType, PoseDim>>());
        if (obs->observations.empty()) return std::move(segmentation);

        std::map<std::array<CellPoseType, POSEDIM>, std::map<size_t, size_t>> counts;
        for (auto i = 0; i < obs->observations.size(); ++i) {
            std::array<CellPoseType, POSEDIM> const pose = ROSTAdapter<POSEDIM, CellPoseType, PoseType>::toCellId(obs->observation_poses[i], this->cell_size);
            size_t const id = unique_obs.get_id(obs->observations[i]);
            if (auto iter = counts.find(pose); iter != counts.end()) {
                if (auto countIter = iter->second.find(id); countIter != iter->second.end()) {
                    countIter->second++;
                } else {
                    iter->second.emplace(id, 1);
                }
            } else {
                counts.emplace(pose, std::map<size_t, size_t>{});
                iter = counts.find(pose);
                if (iter == counts.end()) throw std::logic_error("Failed to insert new pose");
                iter->second.emplace(id, 1);
            }
        }

        if constexpr (std::is_integral_v<LabelType>) {
            // TODO Find max count
        } else if constexpr (is_vector<LabelType>::value) {
            static_assert(std::is_integral_v<typename LabelType::value_type>);
            for (auto const& entry : counts) {
                segmentation->observation_poses.push_back(entry.first);
                LabelType labelVec;
                labelVec.resize(unique_obs.size(), 0);
                for (auto const& idCount : entry.second) {
                    labelVec[idCount.first] = idCount.second;
                }
                segmentation->observations.push_back(labelVec);
            }
        } else {
            static_assert(always_false<LabelType>);
        }

        return segmentation;
    }
};
}

#endif //SUNSHINE_PROJECT_SEGMENTATION_ADAPTER_HPP
