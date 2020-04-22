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

template<typename LabelType, typename CellPoseType = int, typename PoseType = int>
class ImageSegmentationAdapter : public Adapter<ImageSegmentationAdapter<LabelType, CellPoseType, PoseType>, ImageObservation, Segmentation<LabelType, 2, CellPoseType, PoseType>> {
    static size_t constexpr POSEDIM = 2;
    std::array<PoseType, POSEDIM> const cell_size;
    UniqueStore<std::array<uint8_t, 3>> unique_colors;

    std::array<uint8_t, 3> fromVec(cv::Vec3b const& vec) {
        return {vec[0], vec[1], vec[2]};
    }

    template<typename T>
    struct is_vector : public std::false_type {
    };

    template<typename T, typename A>
    struct is_vector<std::vector<T, A>> : public std::true_type {
    };

    template<typename ParameterServer>
    static inline decltype(cell_size) computeCellSize(ParameterServer *nh) {
        double const cell_size_space = nh->template param<double>("cell_space", 100);
        std::string const cell_size_string = nh->template param<std::string>("cell_size", "");
        if (!cell_size_string.empty()) {
            return readNumbers<POSEDIM, 'x', int>(cell_size_string);
        } else {
            return {static_cast<CellPoseType>(cell_size_space), static_cast<CellPoseType>(cell_size_space)};
        }
    }

  public:
    template<typename ParameterServer>
    explicit ImageSegmentationAdapter(ParameterServer *paramServer)
          : cell_size(computeCellSize(paramServer)) {}

    Segmentation<LabelType, POSEDIM, CellPoseType, PoseType> operator()(ImageObservation const &imageObs) {
        typedef Segmentation<LabelType, POSEDIM, CellPoseType, PoseType> Output;
        Output segmentation(imageObs.frame, imageObs.timestamp, imageObs.id, cell_size, {}, {});
        if (imageObs.image.rows == 0 || imageObs.image.cols == 0) return segmentation;

        size_t const numCells = (std::floor(imageObs.image.cols / cell_size[0]) + 1) * (std::floor(imageObs.image.rows / cell_size[1]) + 1);
        segmentation.poses.reserve(numCells);

        std::map<std::array<CellPoseType, POSEDIM>, std::map<size_t, size_t>> counts;
        for (auto y = 0; y < imageObs.image.rows; y++) {
            for (auto x = 0; x < imageObs.image.cols; x++) {
                std::array<CellPoseType, POSEDIM> const pose = ROSTAdapter<2, int, int>::toCellId({x, y}, cell_size);
                size_t const id = unique_colors.get_id(fromVec(imageObs.image.at<cv::Vec3b>(cv::Point(x, y))));
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
            // TODO densify counts
        } else {
            static_assert(always_false<LabelType>);
        }
    }
};
}

#endif //SUNSHINE_PROJECT_SEGMENTATION_ADAPTER_HPP
