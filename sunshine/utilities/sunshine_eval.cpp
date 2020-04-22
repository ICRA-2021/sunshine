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
#include "sunshine/word_depth_adapter.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/ros_conversions.hpp"

template<typename Metric>
double computeMetric(std::string const &bagfile,
                     std::string const &image_topic,
                     std::string const &segmentation_topic,
                     std::string const &depth_topic,
                     sunshine::Parameters const& parameters,
                     Metric const &metric,
                     uint32_t warmup = 0) {
    using namespace sunshine;

    auto visualWordAdapter = VisualWordAdapter(&parameters);
    auto rostAdapter = ROSTAdapter<4, double, double>(&parameters);
    auto segmentationAdapter = ImageSegmentationAdapter<std::vector<size_t>>(&parameters);
    auto depthAdapter = WordDepthAdapter();
    auto transformAdapter = ObservationTransformAdapter<WordDepthAdapter::Output>(&parameters);

    std::unique_ptr<ImageObservation> rgb, segmentation;
    double depth_timestamp = -1;
    auto const processPair = [&](){
        auto wordObs = transformAdapter(depthAdapter(visualWordAdapter(*rgb)));
        auto topicsFuture = rostAdapter(wordObs);
        auto const topicSeg = topicsFuture.get();

        auto const gtSeg = segmentationAdapter(*segmentation);
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

        depthAdapter.updatePointCloud(pc);
        depth_timestamp = msg->header.stamp.toSec();
    };

    auto const transformCallback = [&](tf2_msgs::TFMessage::ConstPtr const& tfMsg){
        for (auto const& transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            transformAdapter.addTransform(tfTransform);
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

    computeMetric(bagfile, image_topic_name, segmentation_topic_name, depth_topic_name, sunshine::Parameters({}), [](){return 0;}, 0);
}
