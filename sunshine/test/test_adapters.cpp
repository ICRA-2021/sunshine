//
// Created by stewart on 3/4/20.
//

#include "sunshine/common/observation_adapters.hpp"
#include "sunshine/rost_adapter.hpp"

using namespace sunshine;

template<typename ImplClass, typename FeatureType, uint32_t PoseDim, typename PoseType = double>
class FeatureExtractorAdapter : public Adapter<ImplClass, ImageObservation, CategoricalObservation<FeatureType, PoseDim, PoseType>> {
};

template<typename ImplClass, typename WordType, typename TopicType, uint32_t PoseDim, typename WordPoseType = double, typename TopicPoseType=WordPoseType>
class TopicModelAdapter : public Adapter<ImplClass, CategoricalObservation<WordType, PoseDim, WordPoseType>, CategoricalObservation<TopicType, PoseDim, TopicPoseType>> {
};

class DummyFeatureExtractor : public FeatureExtractorAdapter<DummyFeatureExtractor, int, 4, double> {
  public:
    std::unique_ptr<Output> operator()(Input const* imgObs) {
        return std::make_unique<Output>(imgObs->frame, imgObs->timestamp, imgObs->id, std::vector<int>{0}, std::vector<std::array<double, 4>>{{0, 10, 20, 30}}, 0, 1);
    }

    using FeatureExtractorAdapter<DummyFeatureExtractor, int, 4, double>::operator();
};

class DummyTopicModel : public TopicModelAdapter<DummyTopicModel, int, int, 4, double, int> {
  public:
    std::unique_ptr<Output> operator()(Input const* wordObs) {
        std::vector<ROSTAdapter<4>::cell_pose_t> cellPoses;
        for (auto const &wordPose : wordObs->observation_poses) cellPoses.push_back(ROSTAdapter<4>::toCellId(wordPose, {10, 10, 10, 10}));
        return std::make_unique<Output>(wordObs->frame,
                                        wordObs->timestamp,
                                        wordObs->id,
                                        wordObs->observations,
                                        cellPoses,
                                        wordObs->vocabulary_start,
                                        wordObs->vocabulary_size);
    }
    using TopicModelAdapter<DummyTopicModel, int, int, 4, double, int>::operator();
};

int main() {
    std::unique_ptr<ImageObservation> dummyImgObs = std::make_unique<ImageObservation>("test", 0x12345678, 0xDEADBEEF, cv::Mat());
    std::cout << "Before: " << std::hex << dummyImgObs->id << std::endl;
    auto output = process<>(std::move(dummyImgObs), DummyFeatureExtractor(), LogAdapter<DummyFeatureExtractor::Output>([](auto const &output) {
        std::cout << "During: " << std::hex << output.id << std::endl;
    }), DummyTopicModel());
    static_assert(std::is_same<decltype(*output), CategoricalObservation<int, 4, int>&>::value, "Processed output has unexpected type");
    std::cout << "After: " << std::hex << output->id << std::endl;
    if (output->frame != dummyImgObs->frame) return 1;
    if (output->timestamp != dummyImgObs->timestamp) return 1;
    if (output->id != dummyImgObs->id) return 1;
    return 0;
}
