//
// Created by stewart on 3/25/20.
//
#ifndef SUNSHINE_PROJECT_SEGMENTATION_ADAPTER_HPP
#define SUNSHINE_PROJECT_SEGMENTATION_ADAPTER_HPP
#include "common/object_stores.hpp"
#include "common/observation_adapters.hpp"
#include "common/utils.hpp"
#include "rost_adapter.hpp"
namespace sunshine {

template<typename ImplClass, typename Input, typename LabelType, uint32_t _PoseDim, typename CellPoseType = int, typename PoseType = int>
class SegmentationAdapter : public Adapter<ImplClass, Input, Segmentation<LabelType, _PoseDim, CellPoseType, PoseType>> {
  protected:
    static size_t constexpr PoseDim = _PoseDim;
    std::array<PoseType, PoseDim> const cell_size;
    template<typename ParameterServer>
    static inline decltype(cell_size) computeCellSize(ParameterServer *nh) {
        double const cell_size_space       = nh->template param<double>("cell_space", sunshine::ROSTAdapter<>::DEFAULT_CELL_SPACE);
        std::string const cell_size_string = nh->template param<std::string>("cell_size", "");
        double cell_size_x = cell_size_space, cell_size_y = cell_size_space, cell_size_z = cell_size_space;
        if (!cell_size_string.empty()) {
            auto cell_size = readNumbers<PoseDim + 1, 'x', PoseType>(cell_size_string);
            cell_size_x = cell_size[1];
            cell_size_y = cell_size[2];
            if constexpr (PoseDim == 3) {
                cell_size_z = cell_size[3];
            }
        }
        if constexpr (PoseDim == 2) {
            return {static_cast<PoseType>(cell_size_x), static_cast<PoseType>(cell_size_y)};
        } else if constexpr (PoseDim == 3) {
            return {static_cast<PoseType>(cell_size_x), static_cast<PoseType>(cell_size_y), static_cast<PoseType>(cell_size_z)};
        } else {
            static_assert(always_false<SegmentationAdapter<ImplClass, Input, LabelType, PoseDim, CellPoseType, PoseType>>);
        }
    }

  public:
    template<typename ParameterServer>
    explicit SegmentationAdapter(ParameterServer *paramServer) : cell_size(computeCellSize(paramServer)) { }
};

template<typename ObservationType, typename LabelType, uint32_t _PoseDim = 3, typename CellPoseType = int, typename PoseType = double>
class SemanticSegmentationAdapter
    : public SegmentationAdapter<SemanticSegmentationAdapter<ObservationType, LabelType, _PoseDim, CellPoseType, PoseType>,
                                 SemanticObservation<ObservationType, _PoseDim, PoseType>,
      LabelType,
      _PoseDim,
      CellPoseType,
      PoseType> {
    static size_t constexpr PoseDim = _PoseDim;
    bool const aggregate;
    UniqueStore<ObservationType> unique_obs;
    std::unordered_map<std::array<CellPoseType, PoseDim>, std::map<size_t, size_t>, hasharray < CellPoseType, PoseDim>> counts;
    std::string frame_id;
    double latest_timestep = 0;
    uint32_t latest_id = 0;

    std::mutex mutable lock;

  public:
    template<typename ParameterServer>
    explicit SemanticSegmentationAdapter(ParameterServer *paramServer, bool aggregate = false)
        : SegmentationAdapter<SemanticSegmentationAdapter<ObservationType, LabelType, _PoseDim, CellPoseType, PoseType>,
                              SemanticObservation<ObservationType, _PoseDim, PoseType>,
                              LabelType,
                              _PoseDim,
                              CellPoseType,
                              PoseType>(paramServer)
        , aggregate(aggregate) { }

    std::unique_ptr<Segmentation<LabelType, PoseDim, CellPoseType, PoseType>> constructSegmentation() const {
        typedef Segmentation<LabelType, PoseDim, CellPoseType, PoseType> Output;
        auto segmentation = std::make_unique<Output>(frame_id, latest_timestep, latest_id, this->cell_size, std::vector<LabelType>(),
            std::vector<std::array<CellPoseType, _PoseDim>>());
        if constexpr (std::is_integral_v<LabelType>) {
            // TODO Find max count
            static_assert(always_false<LabelType>, "Not implemented.");
        } else if constexpr (is_vector<LabelType>::value) {
            static_assert(std::is_integral_v<typename LabelType::value_type>);
            for (auto const &entry : counts) {
                segmentation->observation_poses.push_back(entry.first);
                LabelType labelVec;
                labelVec.resize(unique_obs.size(), 0);
                for (auto const &idCount : entry.second) { labelVec[idCount.first] = idCount.second; }
                segmentation->observations.push_back(labelVec);
            }
        } else {
            static_assert(always_false<LabelType>, "Invalid template argument.");
        }
        //        auto const new_size = counts.size();
        //        std::cout << "Old: " << size << ", New: " << new_size << std::endl;
        return std::move(segmentation);
    }

    [[nodiscard]] std::unique_lock<std::mutex> getLock() const {
        return std::unique_lock(lock);
    }

    auto operator()(SemanticObservation<ObservationType, _PoseDim, PoseType > const *obs) {
        typedef Segmentation<LabelType, PoseDim, CellPoseType, PoseType> Output;
        if (!aggregate) {
            assert(obs != nullptr);
            counts.clear();
        }
        if (obs != nullptr) {
            frame_id = obs->frame;
            latest_timestep = obs->timestamp;
            latest_id = obs->id;
            for (auto i = 0; i < obs->observations.size(); ++i) {
                auto const pose = toCellId<PoseDim, CellPoseType>(obs->observation_poses[i], this->cell_size);
                size_t const id = unique_obs.get_id(obs->observations[i]);
                if (auto iter = counts.find(pose); iter != counts.end()) {
                    if (auto countIter = iter->second.find(id); countIter != iter->second.end()) {
                        countIter->second++;
                    } else {
                        iter->second.emplace(id, 1);
                    }
                } else {
                    counts.emplace(pose, std::map<size_t, size_t>{{id, 1}});
                }
            }
        }
        return constructSegmentation();
    }

    auto operator()(std::unique_ptr<SemanticObservation<ObservationType, _PoseDim, PoseType>> input) {
        return (*this)(input.get());
    }

    auto const& getRawCounts() const {
        return counts;
    }
};
} // namespace sunshine
#endif // SUNSHINE_PROJECT_SEGMENTATION_ADAPTER_HPP
