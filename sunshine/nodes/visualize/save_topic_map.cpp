#include "sunshine/common/data_proc_utils.hpp"
#include "sunshine/common/ros_conversions.hpp"
#include "sunshine/common/utils.hpp"
#include "sunshine/common/word_coloring.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <sunshine_msgs/GetTopicModel.h>
#include <sunshine_msgs/SaveObservationModel.h>
#include <sunshine_msgs/TopicMap.h>

using namespace sunshine_msgs;
using namespace cv;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "save_topic_map");
    ros::NodeHandle nh("~");
    auto const rost_namespace = nh.param<std::string>("topic_model_namespace", "rost");
    double default_pixel_scale = 1;
    try {
        nh.getParam("/" + rost_namespace + "/cell_space", default_pixel_scale);
    } catch (ros::Exception) {
        // do nothing
    }
    auto const pixel_scale = nh.param<double>("pixel_scale", default_pixel_scale);
    auto const input_topic = nh.param<std::string>("input_topic", "/" + rost_namespace + "/topic_map");
    auto const output_prefix = nh.param<std::string>("output_prefix", "topic-map");
    auto const minWidth = nh.param<double>("min_width", 0.);
    auto const minHeight = nh.param<double>("min_height", 0.);
    auto const fixedBox = nh.param<std::string>("box", "");
    auto const useColor = nh.param<bool>("use_color", false);
    auto const continuous = nh.param<bool>("continuous", false);
    auto const saveTopicTimeseries = nh.param<bool>("save_topic_timeseries", false);
    auto const saveTopicModel = nh.param<bool>("save_topic_model", true);
    auto const saveTopicCells = nh.param<bool>("save_topic_cells", true);
    auto const savePpx = nh.param<bool>("save_perplexity_map", true);
    sunshine::WordColorMap<decltype(TopicMap::cell_topics)::value_type> wordColorMap;
    ROS_INFO("Pixel scale: %f", pixel_scale);
    bool done = false;

    ros::ServiceClient client = nh.serviceClient<SaveObservationModel>("/" + rost_namespace + "/save_topics_by_time_csv");
    ros::ServiceClient cellClient = nh.serviceClient<SaveObservationModel>("/" + rost_namespace + "/save_topics_by_cell_csv");
    ros::ServiceClient modelClient = nh.serviceClient<GetTopicModel>("/" + rost_namespace + "/get_topic_model");

    auto obsSub = nh.subscribe<TopicMap>(input_topic, 1, [&done, &wordColorMap, &client, &modelClient, &cellClient, output_prefix, pixel_scale, minWidth,
                                                          savePpx, saveTopicTimeseries, saveTopicModel, saveTopicCells, minHeight, useColor, fixedBox](sunshine_msgs::TopicMapConstPtr const& msg) {
        if (saveTopicModel) {
            GetTopicModel getTopicModel;
            if (modelClient.call(getTopicModel)) {
                std::string const filename = output_prefix + "-" + std::to_string(msg->seq) + "-modelweights.bin";
                try {
                    std::ofstream writer(filename, std::ios::out | std::ios::binary);
                    sunshine::fromRosMsg(getTopicModel.response.topic_model).serialize(writer);
                } catch (std::logic_error const& e) {
                    ROS_ERROR("Failed to save topic model to file %s", filename.c_str());
                }
            } else {
                ROS_ERROR("Failed to save topic model!");
            }
        }

        if (saveTopicTimeseries) {
            SaveObservationModel saveObservationModel;
            saveObservationModel.request.filename = output_prefix + "-" + std::to_string(msg->seq) + "-timeseries.csv";
            if (client.call(saveObservationModel)) {
                ROS_INFO("Saved timeseries to %s", saveObservationModel.request.filename.c_str());
            } else {
                ROS_ERROR("Failed to save topic timeseries to %s!", saveObservationModel.request.filename.c_str());
            }
        }

        if (saveTopicCells) {
            SaveObservationModel saveObservationModel;
            saveObservationModel.request.filename = output_prefix + "-" + std::to_string(msg->seq) + "-cells.csv";
            if (cellClient.call(saveObservationModel)) {
                ROS_INFO("Saved timeseries to %s", saveObservationModel.request.filename.c_str());
            } else {
                ROS_ERROR("Failed to save topic timeseries to %s!", saveObservationModel.request.filename.c_str());
            }
        }

        Mat topicMapImg = sunshine::createTopicImg(*msg, wordColorMap, pixel_scale, useColor, minWidth, minHeight, fixedBox, true);
        sunshine::saveTopicImg(topicMapImg, output_prefix + "-" + std::to_string(msg->seq) + "-topics.png", output_prefix + "-" + std::to_string(msg->seq) + "-colors.csv", &wordColorMap);
        done = true;
    });

    if (continuous) {
        ros::spin();
    } else {
        ros::Rate idler(10);
        while (!done) {
            ros::spinOnce();
            idler.sleep();
        }
    }
}
