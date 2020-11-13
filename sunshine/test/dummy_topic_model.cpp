//
// Created by stewart on 2020-07-14.
//

#include "sunshine/common/ros_utils.hpp"
#include <exception>
#include <sunshine_msgs/GetTopicModel.h>
#include <sunshine_msgs/SetTopicModel.h>
#include <sunshine_msgs/Pause.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"

using namespace sunshine;
using namespace sunshine_msgs;

bool get_topic_model(GetTopicModelRequest &, GetTopicModelResponse &response) {
    return false;
}

bool set_topic_model(SetTopicModelRequest &request, SetTopicModelResponse &) {
    return false;
}

bool pause_topic_model(PauseRequest &request, PauseResponse &) {
    ROS_DEBUG("Changing topic model global pause state");
    return true;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "topic_model");
    ros::NodeHandle nh("~");

    ros::ServiceServer get_topic_model_server, set_topic_model_server, pause_server;
    get_topic_model_server = nh.advertiseService<>("get_topic_model", get_topic_model);
    set_topic_model_server = nh.advertiseService<>("set_topic_model", set_topic_model);
    pause_server = nh.advertiseService<>("pause_topic_model", pause_topic_model);

    ros::MultiThreadedSpinner spinner;
    ROS_INFO("Spinning...");
    spinner.spin();
}