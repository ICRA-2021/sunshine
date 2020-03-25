//
// Created by stewart on 3/9/20.
//

#include "../include/sunshine/word_depth_adapter.hpp"

#include <utility>

using namespace sunshine;

CategoricalObservation<int, 3, double> WordDepthAdapter::operator()(CategoricalObservation<int, 2, int> &&wordObs2d) const {
    if (!pc) throw std::logic_error("Cannot process input without pointcloud");
    auto const num_words = wordObs2d.observations.size();

    std::vector<std::array<double, 3>> observation_pose;
    observation_pose.reserve(num_words * 4);
    for (size_t i = 0; i < num_words; ++i) {
        int u, v;
        u = wordObs2d.observation_poses[i][0];
        v = wordObs2d.observation_poses[i][1];
        auto const &cloud = *pc;
        assert(u < cloud.width && v < cloud.height);
        auto const pcPose = cloud.at(u, v).getArray3fMap();
        observation_pose.push_back({pcPose.x(), pcPose.y(), pcPose.z()});
    }

    return CategoricalObservation<int, 3, double>(wordObs2d.frame,
                                                  wordObs2d.timestamp,
                                                  wordObs2d.id,
                                                  std::move(wordObs2d.observations),
                                                  std::move(observation_pose),
                                                  wordObs2d.vocabulary_start,
                                                  wordObs2d.vocabulary_size);
}

void WordDepthAdapter::updatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr newPc) {
    this->pc = std::move(newPc);
}
