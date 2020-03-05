#include "utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using namespace sunshine;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sunshine_eval");

    ros::NodeHandle nh("~");
    auto const ground_truth_topic = nh.param<std::string>("ground_truth", argv[1]);
    auto const topics_topic = nh.param<std::string>("topic_labels", argv[2]);
    auto const cell_size_string = nh.param<std::string>("cell_size", "");
    auto const cell_size_space = nh.param<double>("cell_space", 1);
    auto const cell_size_time = nh.param<double>("cell_time", 1);


    while (nh.ok()) {
        ros::spin();
    }
}
