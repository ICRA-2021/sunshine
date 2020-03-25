#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <utility>
#include "sunshine/rost_adapter.hpp"
#include "sunshine/visual_word_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/ros_conversions.hpp"

template<typename Metric>
double computeMetric(std::string const &bagfile,
                     std::string const &image_topic,
                     std::string const &segmentation_topic,
                     sunshine::Parameters const& parameters,
                     Metric const &metric,
                     uint32_t warmup = 0) {
    using namespace sunshine;

    auto visualWordAdapter = VisualWordAdapter(&parameters);
    auto rostAdapter = ROSTAdapter<3, int, double>(&parameters);
    auto segmentationAdapter = ImageSegmentationAdapter<std::vector<size_t>>(&parameters);

    std::unique_ptr<ImageObservation> rgb, segmentation, depth;
    auto const processPair = [&](){
        auto wordObs = visualWordAdapter(*rgb);
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
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp) {
            processPair();
        }
    };

    auto const depthCallback = [&](sensor_msgs::Image::ConstPtr image){
        depth = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && depth && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth->timestamp) {
            processPair();
        }
    };

    ROS_WARN("Extracting images from %s", bagfile.c_str());
    rosbag::Bag bag;
    bag.open(bagfile);

    size_t i = 0;
    for (rosbag::MessageInstance const m: rosbag::View(bag)) {
        if (m.getTopic() == image_topic || m.getTopic() == segmentation_topic) {
            sensor_msgs::Image::ConstPtr imgMsg = m.instantiate<sensor_msgs::Image>();
            if (imgMsg != nullptr) {
                if (m.getTopic() == image_topic) imageCallback(imgMsg);
                if (m.getTopic() == segmentation_topic) segmentationCallback(imgMsg);
//                if (m.getTopic() == depth_topic) depthCallback(imgMsg);
            } else {
                ROS_ERROR("Non-image message found in topic %s", m.getTopic().c_str());
            }
        }
    }
    bag.close();
    ROS_INFO("Extracted %lu images from rosbag.", i);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./sunshine_eval bagfile image_topic segmentation_topic" << std::endl;
        return 1;
    }
    std::string const bagfile(argv[1]);
    std::string const image_topic_name(argv[2]);
    std::string const segmentation_topic_name(argv[3]);

    computeMetric(bagfile, image_topic_name, segmentation_topic_name, sunshine::Parameters({}), [](){return 0;}, 0);
}
