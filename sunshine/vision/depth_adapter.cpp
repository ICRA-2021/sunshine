//
// Created by stewart on 3/9/20.
//

#include "sunshine/depth_adapter.hpp"

#include <utility>

using namespace sunshine;

void DepthAdapter::updatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc) {
    this->pc = std::move(pc);
}

std::array<double, 3> inline DepthAdapter::get_pose(int u, int v) const {
    if (!pc) throw std::logic_error("Cannot process input without pointcloud");
    auto const &cloud = *pc;
    assert(u < cloud.width && v < cloud.height);
    auto const pcPose = cloud.at(u, v).getArray3fMap();
    return {pcPose.x(), pcPose.y(), pcPose.z()};
}

std::unique_ptr<CategoricalObservation<int, 3, double>> WordDepthAdapter::operator()(CategoricalObservation<int, 2, int> const * const wordObs2d) const {
    auto const num_words = wordObs2d->observations.size();

    std::vector<std::array<double, 3>> observation_pose;
    observation_pose.reserve(num_words * 4);
    for (size_t i = 0; i < num_words; ++i) {
        int const u = wordObs2d->observation_poses[i][0], v = wordObs2d->observation_poses[i][1];
        observation_pose.push_back(this->get_pose(u, v));
    }

    return std::make_unique<CategoricalObservation<int, 3, double>>(wordObs2d->frame,
                                                                    wordObs2d->timestamp,
                                                                    wordObs2d->id,
                                                                    wordObs2d->observations,
                                                                    std::move(observation_pose),
                                                                    wordObs2d->vocabulary_start,
                                                                    wordObs2d->vocabulary_size);
}

std::unique_ptr<SemanticObservation<std::array<uint8_t, 3>, 3, double>> ImageDepthAdapter::operator()(ImageObservation const * const wordObs2d) const {
    auto const num_words = wordObs2d->image.rows * wordObs2d->image.cols;

    std::vector<std::array<uint8_t, 3>> observations;
    std::vector<std::array<double, 3>> observation_pose;
    observation_pose.reserve(num_words * 4);
    for (int y = 0; y < wordObs2d->image.rows; ++y) {
        for (int x = 0; x < wordObs2d->image.cols; ++x) {
            auto const& rgb = wordObs2d->image.at<cv::Vec3b>(cv::Point(x, y));
            observations.push_back({rgb[0], rgb[1], rgb[2]});
            observation_pose.push_back(this->get_pose(x, y));
        }
    }

    return std::make_unique<SemanticObservation<std::array<uint8_t, 3>, 3, double>>(wordObs2d->frame,
                                                                    wordObs2d->timestamp,
                                                                    wordObs2d->id,
                                                                    std::move(observations),
                                                                    std::move(observation_pose));
}
