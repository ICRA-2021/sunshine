//
// Created by stewart on 3/9/20.
//

#ifndef SUNSHINE_PROJECT_DEPTH_ADAPTER_HPP
#define SUNSHINE_PROJECT_DEPTH_ADAPTER_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <array>
#include "sunshine/common/observation_adapters.hpp"

namespace sunshine {

class DepthAdapter {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc;

  protected:
    [[nodiscard]] std::array<double, 3> get_pose(int x, int y) const;
  public:
    void updatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc);
};

class WordDepthAdapter : public Adapter<WordDepthAdapter, CategoricalObservation<int, 2, int>, CategoricalObservation<int, 3, double>>, public DepthAdapter {
  public:
    WordDepthAdapter() = default;
    std::unique_ptr<CategoricalObservation<int, 3, double>> operator()(std::unique_ptr<CategoricalObservation<int, 2, int>> const& wordObs2d) const;
};

class ImageDepthAdapter : public Adapter<ImageDepthAdapter, ImageObservation, SemanticObservation<std::array<uint8_t, 3>, 3, double>>, public DepthAdapter {
    public:
    ImageDepthAdapter() = default;
    std::unique_ptr<SemanticObservation<std::array<uint8_t, 3>, 3, double>> operator()(std::unique_ptr<ImageObservation> const& image) const;
};
}


#endif //SUNSHINE_PROJECT_DEPTH_ADAPTER_HPP
