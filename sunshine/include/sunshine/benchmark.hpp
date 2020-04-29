//
// Created by stewart on 4/28/20.
//

#ifndef SUNSHINE_PROJECT_BENCHMARK_HPP
#define SUNSHINE_PROJECT_BENCHMARK_HPP

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf/tf.h>
#include <tf2_msgs/TFMessage.h>
#include <tf/transform_datatypes.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>

#include <utility>
#include <set>

#include "sunshine/rost_adapter.hpp"
#include "sunshine/visual_word_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"
#include "sunshine/observation_transform_adapter.hpp"
#include "sunshine/depth_adapter.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/ros_conversions.hpp"

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
//        auto wordObs = wordTransformAdapter(wordDepthAdapter(visualWordAdapter(rgb.get())));
        auto topicsFuture = rgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;

//        auto const segObs = imageTransformAdapter(imageDepthAdapter(segmentation));
        auto gtSeg = segmentation >> imageDepthAdapter >> imageTransformAdapter >> segmentationAdapter;
        auto topicsSeg = topicsFuture.get();
        if (count >= warmup) {
            double const result = metric(*gtSeg, *topicsSeg);
            av_metric = (av_metric * (count - warmup) + result) / (count - warmup + 1);
        }
        count += 1;
    };

    auto const imageCallback = [&](sensor_msgs::Image::ConstPtr image) {
        rgb = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            processPair();
        }
    };

    auto const segmentationCallback = [&](sensor_msgs::Image::ConstPtr image) {
        segmentation = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            processPair();
        }
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
            processPair();
        }
    };

    auto const transformCallback = [&](tf2_msgs::TFMessage::ConstPtr const &tfMsg) {
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
        }
    };

    ROS_WARN("Extracting images from %s", bagfile.c_str());
    rosbag::Bag bag;
    bag.open(bagfile);

    std::set<std::string> skipped_topics;
    for (rosbag::MessageInstance const m: rosbag::View(bag)) {
        if (m.getTopic() == image_topic || m.getTopic() == segmentation_topic) {
            sensor_msgs::Image::ConstPtr imgMsg = m.instantiate<sensor_msgs::Image>();
            if (imgMsg != nullptr) {
                if (m.getTopic() == image_topic) imageCallback(imgMsg);
                if (m.getTopic() == segmentation_topic) segmentationCallback(imgMsg);
            } else {
                ROS_ERROR("Non-image message found in topic %s", m.getTopic().c_str());
            }
        } else if (m.getTopic() == depth_topic) {
            depthCallback(m.instantiate<sensor_msgs::PointCloud2>());
        } else if (m.getTopic() == "/tf") {
            transformCallback(m.instantiate<tf2_msgs::TFMessage>());
        } else {
            if (skipped_topics.find(m.getTopic()) == skipped_topics.end()) {
                ROS_INFO("Skipped message from topic %s", m.getTopic().c_str());
                skipped_topics.insert(m.getTopic());
            }
        }
        if (count >= max_iter) break;
    }
    bag.close();
    ROS_INFO("Processed %u images from rosbag.", count);
    return av_metric;
}

template<typename Container>
double entropy(Container const &container, double weight = 1.0) {
    double sum = 0;
    for (auto const &val : container) {
        if (val > 0) {
            double const pv = (weight == 1.0)
                              ? val
                              : (val / weight);
            sum += pv * log(pv);
        } else {
            assert(val == 0);
        }
    }
    return -sum;
}

template<typename Container>
uint32_t argmax(Container const &container) {
    uint32_t idx_max = 0;
    auto max = container[0];
    for (auto i = 1; i < container.size(); ++i) {
        if (container[i] > max) {
            idx_max = i;
            max = container[i];
        }
    }
    return idx_max;
}

double nmi(sunshine::Segmentation<std::vector<int>, 3, int, double> const &gt_seg,
           sunshine::Segmentation<std::vector<int>, 4, int, double> const &topic_seg) {
    std::map<std::array<int, 3>, uint32_t> gt_labels, topic_labels;
    std::vector<uint32_t> gt_weights(gt_seg.observations[0].size(), 0), topic_weights(topic_seg.observations[0].size(), 0);
    std::vector<std::vector<uint32_t>> matches(topic_seg.observations[0].size(), std::vector<uint32_t>(gt_seg.observations[0].size(), 0));
    double total_weight = 0;
    for (auto i = 0; i < gt_seg.observations.size(); ++i) {
        auto const label = argmax<>(gt_seg.observations[i]);
        gt_labels.insert({gt_seg.observation_poses[i], label});
    }
    for (auto i = 0; i < topic_seg.observations.size(); ++i) {
        std::array<int, 3> pose{topic_seg.observation_poses[i][1], topic_seg.observation_poses[i][2], topic_seg.observation_poses[i][3]};
        auto const topic_label = argmax<>(topic_seg.observations[i]);
        topic_labels.insert({std::move(pose), topic_label});

        auto iter = gt_labels.find(pose);
        if (iter != gt_labels.end()) {
            auto const &gt_label = iter->second;
            matches[topic_label][gt_label] += 1;
            gt_weights[gt_label] += 1;
            topic_weights[topic_label] += 1;
            total_weight += 1;
        } else {
            std::cerr << "Failed to find gt gt_seg for pose!" << std::endl;
        }
    }

    double mi = 0;
    double sum_pxy = 0;
    for (auto i = 0; i < matches.size(); ++i) {
        for (auto j = 0; j < matches[i].size(); ++j) {
            auto const px = topic_weights[i] / total_weight;
            auto const py = gt_weights[j] / total_weight;
            auto const pxy = matches[i][j] / total_weight;
            if (pxy > 0) {
                mi += pxy * log(pxy / (px * py));
            }
            sum_pxy += pxy;
        }
    }

    double const ex = entropy<>(gt_weights, total_weight), ey = entropy<>(topic_weights, total_weight);
    double const nmi = (mi == 0)
                       ? 0.
                       : (mi / sqrt(ex * ey));

    return nmi;
}
}

#endif //SUNSHINE_PROJECT_BENCHMARK_HPP
