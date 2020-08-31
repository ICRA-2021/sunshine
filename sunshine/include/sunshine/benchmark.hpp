//
// Created by stewart on 4/28/20.
//

#ifndef SUNSHINE_PROJECT_BENCHMARK_HPP
#define SUNSHINE_PROJECT_BENCHMARK_HPP

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <tf2_msgs/TFMessage.h>
#include <tf/transform_datatypes.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <utility>

#include <set>
#include <cmath>
#include "sunshine/rost_adapter.hpp"

#include "sunshine/visual_word_adapter.hpp"
#include "sunshine/semantic_label_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"
#include "sunshine/observation_transform_adapter.hpp"
#include "sunshine/depth_adapter.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/ros_conversions.hpp"
#include "sunshine/common/rosbag_utils.hpp"

namespace sunshine {
template<typename Metric>
double benchmark(std::string const &bagfile,
                 std::string const &image_topic,
                 std::string const &segmentation_topic,
                 std::string const &depth_topic,
                 sunshine::Parameters const &parameters,
                 Metric const &metric,
                 uint32_t const warmup = 0,
                 uint32_t const max_iter = std::numeric_limits<uint32_t>::max()) {
    using namespace sunshine;

    auto visualWordAdapter = VisualWordAdapter(&parameters);
//    auto labelSegmentationAdapter = SemanticSegmentationAdapter<int, std::vector<int>>(&parameters);
    auto rostAdapter = ROSTAdapter<4, double, double>(&parameters);
    auto segmentationAdapter = SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>(&parameters);
    auto wordDepthAdapter = WordDepthAdapter();
    auto imageDepthAdapter = ImageDepthAdapter();
    auto wordTransformAdapter = ObservationTransformAdapter<WordDepthAdapter::Output>(&parameters);
    auto imageTransformAdapter = ObservationTransformAdapter<ImageDepthAdapter::Output>(&parameters);
    uint32_t count = 0;
    double av_metric = 0;

    std::unique_ptr<ImageObservation> rgb, segmentation;
    double depth_timestamp = -1;
    auto const processPair = [&]() {
        auto topicsFuture = rgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;
        auto gtSeg = segmentation >> imageDepthAdapter >> imageTransformAdapter >> segmentationAdapter;
        auto topicsSeg = topicsFuture.get();

        if (count >= warmup) {
            double const result = metric(*gtSeg, *topicsSeg);
            av_metric = (av_metric * (count - warmup) + result) / (count - warmup + 1);
        }
        count += 1;
        return count >= max_iter;
    };

    auto const imageCallback = [&](sensor_msgs::Image::ConstPtr image) {
        rgb = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            return processPair();
        }
        return false;
    };

    auto const segmentationCallback = [&](sensor_msgs::Image::ConstPtr image) {
        segmentation = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            return processPair();
        }
        return false;
    };

    auto const depthCallback = [&](sensor_msgs::PointCloud2::ConstPtr const &msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pc);

        wordDepthAdapter.updatePointCloud(pc);
        imageDepthAdapter.updatePointCloud(pc);
        depth_timestamp = msg->header.stamp.toSec();
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            return processPair();
        }
        return false;
    };

    auto const transformCallback = [&](tf2_msgs::TFMessage::ConstPtr const &tfMsg) {
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
        }
        return false;
    };

    ROS_WARN("Extracting images from %s", bagfile.c_str());
    sunshine::BagIterator bagIter(bagfile);
    bagIter.add_callback<sensor_msgs::Image>(image_topic, imageCallback);
    bagIter.add_callback<sensor_msgs::Image>(segmentation_topic, segmentationCallback);
    bagIter.add_callback<sensor_msgs::PointCloud2>(depth_topic, depthCallback);
    bagIter.add_callback<tf2_msgs::TFMessage>("/tf", transformCallback);
    bagIter.set_logging(true);
    auto const finished = bagIter.play();
    ROS_ERROR_COND(!finished, "Failed to finish playing bagfile!");
    ROS_INFO("Processed %u images from rosbag.", count);
    return av_metric;
}
}

#endif //SUNSHINE_PROJECT_BENCHMARK_HPP
