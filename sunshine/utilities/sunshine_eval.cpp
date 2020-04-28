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

template<typename Metric>
double computeMetric(std::string const &bagfile,
                     std::string const &image_topic,
                     std::string const &segmentation_topic,
                     std::string const &depth_topic,
                     sunshine::Parameters const& parameters,
                     Metric const &metric,
                     uint32_t const warmup = 0) {
    using namespace sunshine;

    auto visualWordAdapter = VisualWordAdapter(&parameters);
    auto rostAdapter = ROSTAdapter<4, double, double>(&parameters);
    auto segmentationAdapter = SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>(&parameters);
    auto wordDepthAdapter = WordDepthAdapter();
    auto imageDepthAdapter = ImageDepthAdapter();
    auto wordTransformAdapter = ObservationTransformAdapter<WordDepthAdapter::Output>(&parameters);
    auto imageTransformAdapter = ObservationTransformAdapter<ImageDepthAdapter::Output>(&parameters);
    uint32_t count = 0;

    std::unique_ptr<ImageObservation> rgb, segmentation;
    double depth_timestamp = -1;
    auto const processPair = [&](){
        auto wordObs = wordTransformAdapter(wordDepthAdapter(visualWordAdapter(rgb)));
        auto topicsFuture = rostAdapter(wordObs);

        auto const segObs = imageTransformAdapter(imageDepthAdapter(segmentation));
        auto gtSeg = segmentationAdapter(segObs);
        auto topicsSeg = topicsFuture.get();
        if (count++ >= warmup) {
            metric(*gtSeg, *topicsSeg);
        }
    };

    auto const imageCallback = [&](sensor_msgs::Image::ConstPtr image){
        rgb = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp) {
            processPair();
        }
    };

    auto const segmentationCallback = [&](sensor_msgs::Image::ConstPtr image){
        segmentation = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp <= depth_timestamp) {
            processPair();
        }
    };

    auto const depthCallback = [&](sensor_msgs::PointCloud2::ConstPtr const& msg){
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pc);

        wordDepthAdapter.updatePointCloud(pc);
        imageDepthAdapter.updatePointCloud(pc);
        depth_timestamp = msg->header.stamp.toSec();
    };

    auto const transformCallback = [&](tf2_msgs::TFMessage::ConstPtr const& tfMsg){
        for (auto const& transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
        }
    };

    ROS_WARN("Extracting images from %s", bagfile.c_str());
    rosbag::Bag bag;
    bag.open(bagfile);

    size_t i = 0;
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
    }
    bag.close();
    ROS_INFO("Extracted %lu images from rosbag.", i);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: ./sunshine_eval bagfile image_topic depth_topic segmentation_topic" << std::endl;
        return 1;
    }
    std::string const bagfile(argv[1]);
    std::string const image_topic_name(argv[2]);
    std::string const depth_topic_name(argv[3]);
    std::string const segmentation_topic_name(argv[4]);

    computeMetric(bagfile, image_topic_name, segmentation_topic_name, depth_topic_name, sunshine::Parameters({}), [](sunshine::Segmentation<std::vector<int>, 3, int, double> const& labels, sunshine::Segmentation<std::vector<int>, 4, int, double> const& seg){
        std::map<std::array<int, 3>, uint32_t> gt_labels, topic_labels;
        std::vector<uint32_t> gt_weights(seg.observations[0].size(), 0), topic_weights(labels.observations[0].size(), 0);
        std::vector<std::vector<uint32_t>> matches(seg.observations[0].size(), std::vector<uint32_t>(labels.observations[0].size(), 0));
        double total_weight = 0;
        auto const argmax = [](auto container){
            uint32_t idx_max = 0;
            auto max = container[0];
            for (auto i = 1; i < container.size(); ++i) {
                if (container[i] > max) {
                    idx_max = i;
                    max = container[i];
                }
            }
            return idx_max;
        };
        for (auto i = 0; i < seg.observations.size(); ++i) {
            std::array<int, 3> pose{seg.observation_poses[i][1], seg.observation_poses[i][2], seg.observation_poses[i][3]};
            auto const label = argmax(seg.observations[i]);
            gt_labels.insert({std::move(pose), label});
        }
        for (auto i = 0; i < labels.observations.size(); ++i) {
            auto const label = argmax(labels.observations[i]);
            topic_labels.insert({labels.observation_poses[i], label});

            auto iter = gt_labels.find(labels.observation_poses[i]);
            if (iter != gt_labels.end()) {
                matches[iter->second][label] += 1;
                gt_weights[iter->second] += 1;
                topic_weights[label] += 1;
                total_weight += 1;
            } else {
                std::cerr << "Failed to find gt labels for pose!" << std::endl;
            }
        }

        auto const entropy = [](auto container, double weight = 1.0){
            double sum = 0;
            for (auto const& val : container) {
                if (val > 0) {
                    double const pv = (weight == 1.0) ? val : (val / weight);
                    sum += pv * log(pv);
                } else {
                    assert(val == 0);
                }
            }
            return -sum;
        };

        double mi = 0;
        double sum_pxy = 0;
        for (auto i = 0; i < matches.size(); ++i) {
            for (auto j = 0; j < matches[i].size(); ++j) {
                auto const px = gt_weights[i] / total_weight;
                auto const py = topic_weights[j] / total_weight;
                auto const pxy = matches[i][j] / total_weight;
                if (pxy > 0) {
                    mi += pxy * log(pxy / (px * py));
                }
                sum_pxy += pxy;
            }
        }

        double const ex = entropy(gt_weights, total_weight), ey = entropy(topic_weights, total_weight);
        double const nmi = (mi == 0) ? 0. : (mi / sqrt(ex * ey));

        return nmi;
    }, 10);
}
