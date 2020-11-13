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

template<typename ImplClass, typename Input, typename LabelType, uint32_t PoseDim, typename CellPoseType = int, typename PoseType = int>
class SegmentationAdapter : public Adapter<ImplClass, Input, Segmentation<LabelType, PoseDim, CellPoseType, PoseType>> {
  protected:
    static size_t constexpr POSEDIM = PoseDim;
    std::array<PoseType, POSEDIM> const cell_size;
    template<typename ParameterServer>
    static inline decltype(cell_size) computeCellSize(ParameterServer *nh) {
        double const cell_size_space       = nh->template param<double>("cell_space", sunshine::ROSTAdapter<>::DEFAULT_CELL_SPACE);
        std::string const cell_size_string = nh->template param<std::string>("cell_size", "");
        double cell_size_x = cell_size_space, cell_size_y = cell_size_space, cell_size_z = cell_size_space;
        if (!cell_size_string.empty()) {
            auto cell_size = readNumbers<POSEDIM + 1, 'x', PoseType>(cell_size_string);
            cell_size_x = cell_size[1];
            cell_size_y = cell_size[2];
            if constexpr (POSEDIM == 3) {
                cell_size_z = cell_size[3];
            }
        }
        if constexpr (POSEDIM == 2) {
            return {static_cast<PoseType>(cell_size_x), static_cast<PoseType>(cell_size_y)};
        } else if constexpr (POSEDIM == 3) {
            return {static_cast<PoseType>(cell_size_x), static_cast<PoseType>(cell_size_y), static_cast<PoseType>(cell_size_z)};
        } else {
            static_assert(always_false<SegmentationAdapter<ImplClass, Input, LabelType, PoseDim, CellPoseType, PoseType>>);
        }
    }

  public:
    template<typename ParameterServer>
    explicit SegmentationAdapter(ParameterServer *paramServer) : cell_size(computeCellSize(paramServer)) { }
};

template<typename ObservationType, typename LabelType, uint32_t PoseDim = 3, typename CellPoseType = int, typename PoseType = double>
class SemanticSegmentationAdapter
    : public SegmentationAdapter<SemanticSegmentationAdapter<ObservationType, LabelType, PoseDim, CellPoseType, PoseType>,
                                 SemanticObservation<ObservationType, PoseDim, PoseType>,
                                 LabelType,
                                 PoseDim,
                                 CellPoseType,
                                 PoseType> {
    static size_t constexpr POSEDIM = PoseDim;
    bool const aggregate;
    UniqueStore<ObservationType> unique_obs;
    std::unordered_map<std::array<CellPoseType, POSEDIM>, std::map<size_t, size_t>, hasharray<CellPoseType, POSEDIM>> counts;

  public:
    template<typename ParameterServer>
    explicit SemanticSegmentationAdapter(ParameterServer *paramServer, bool aggregate = false)
        : SegmentationAdapter<SemanticSegmentationAdapter<ObservationType, LabelType, PoseDim, CellPoseType, PoseType>,
                              SemanticObservation<ObservationType, PoseDim, PoseType>,
                              LabelType,
                              PoseDim,
                              CellPoseType,
                              PoseType>(paramServer)
        , aggregate(aggregate) { }

    auto operator()(SemanticObservation<ObservationType, PoseDim, PoseType> const *obs) {
        typedef Segmentation<LabelType, POSEDIM, CellPoseType, PoseType> Output;
        if (!aggregate) { counts.clear(); }
        std::unique_ptr<Segmentation<LabelType, POSEDIM, CellPoseType, PoseType>> segmentation =
            std::make_unique<Output>(obs->frame, obs->timestamp, obs->id, this->cell_size, std::vector<LabelType>(),
                                     std::vector<std::array<CellPoseType, PoseDim>>());
//        auto const size = counts.size();
        if (obs->observations.empty()) return std::move(segmentation);
        for (auto i = 0; i < obs->observations.size(); ++i) {
            auto const pose = toCellId<POSEDIM, CellPoseType>(obs->observation_poses[i], this->cell_size);
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
        return segmentation;
    }

    auto operator()(std::unique_ptr<SemanticObservation<ObservationType, PoseDim, PoseType>> input) {
        return (*this)(input.get());
    }

    auto const& getRawCounts() const {
        return counts;
    }
};
} // namespace sunshine
#endif // SUNSHINE_PROJECT_SEGMENTATION_ADAPTER_HPP