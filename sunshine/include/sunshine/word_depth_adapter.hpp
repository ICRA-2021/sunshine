//
// Created by stewart on 3/9/20.
//

#ifndef SUNSHINE_PROJECT_WORD_DEPTH_ADAPTER_HPP
#define SUNSHINE_PROJECT_WORD_DEPTH_ADAPTER_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "sunshine/common/observation_adapters.hpp"

namespace sunshine {
class WordDepthAdapter : public Adapter<WordDepthAdapter, CategoricalObservation<int, 2, int>, CategoricalObservation<int, 3, double>> {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc;

  public:
    WordDepthAdapter() = default;
    CategoricalObservation<int, 3, double> operator()(CategoricalObservation<int, 2, int> &&wordObs2d) const;

    void updatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc);
};
}

#endif //SUNSHINE_PROJECT_WORD_DEPTH_ADAPTER_HPP
