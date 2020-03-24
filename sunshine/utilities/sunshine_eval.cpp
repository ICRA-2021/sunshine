#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "sunshine/rost_adapter.hpp"
#include "sunshine/visual_word_adapter.hpp"
#include "sunshine/word_depth_adapter.hpp"

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./sunshine_eval bagfile image_topic segmentation_topic" << std::endl;
        return 1;
    }
    std::string const bagfile(argv[1]);
    std::string const image_topic_name(argv[2]);
    std::string const segmentation_topic_name(argv[3]);

    using namespace sunshine;
    ROSTAdapter rostAdapter()

    ROS_WARN("Extracting images from %s", bagfile.c_str());
    rosbag::Bag bag;
    bag.open(bagfile);

    size_t i = 0;
    for (rosbag::MessageInstance const m: rosbag::View(bag)) {
        if (m.getTopic() == image_topic_name) {
            sensor_msgs::Image::ConstPtr imgMsg = m.instantiate<sensor_msgs::Image>();
            if (imgMsg != nullptr) {
                imageCallback(imgMsg);
            } else {
                ROS_ERROR("Non-image message found in topic %s", image_topic_name.c_str());
            }
        }
    }
    bag.close();
    ROS_INFO("Extracted %lu images from rosbag.", i);
}
