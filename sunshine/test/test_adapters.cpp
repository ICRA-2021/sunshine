//
// Created by stewart on 3/4/20.
//

#include "sunshine/common/observation_adapters.hpp"
#include "sunshine/rost_adapter.hpp"

using namespace sunshine;

class DummyFeatureExtractor : public FeatureExtractorAdapter<DummyFeatureExtractor, int, 4, double> {
  public:
    Output operator()(const Input& imgObs) {
        return Output(imgObs.frame, imgObs.timestamp, imgObs.id, {0}, {{0, 10, 20, 30}}, 0, 1);
    }
};

class DummyTopicModel : public TopicModelAdapter<DummyTopicModel, int, int, 4, double, int> {
  public:
    Output operator()(const Input& wordObs) {
        std::vector<ROSTAdapter<4>::cell_pose_t> cellPoses;
        for (auto const &wordPose : wordObs.observation_poses) cellPoses.push_back(ROSTAdapter<4>::toCellId(wordPose, {10, 10, 10, 10}));
        return Output(wordObs.frame,
                      wordObs.timestamp,
                      wordObs.id,
                      wordObs.observations,
                      cellPoses,
                      wordObs.vocabulary_start,
                      wordObs.vocabulary_size);
    }
};

int main() {
    ImageObservation dummyImgObs("test", 0x12345678, 0xDEADBEEF, {});
    std::cout << "Before: " << std::hex << dummyImgObs.id << std::endl;
    auto output = process<>(dummyImgObs, DummyFeatureExtractor(), LogAdapter<DummyFeatureExtractor::Output>([](auto const& output){
        std::cout << "During: " << std::hex << output.id << std::endl;
    }), DummyTopicModel());
    static_assert(std::is_same<decltype(output), CategoricalObservation<int, 4, int>>::value, "Processed output has unexpected type");
    std::cout << "After: " << std::hex << output.id << std::endl;
    if (output.frame != dummyImgObs.frame) return 1;
    if (output.timestamp != dummyImgObs.timestamp) return 1;
    if (output.id != dummyImgObs.id) return 1;
    return 0;
}
