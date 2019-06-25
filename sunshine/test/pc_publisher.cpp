// From http://wiki.ros.org/image_transport/Tutorials/PublishingImages

#include "image_utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pc_publisher");

    ros::NodeHandle nh("~");
    auto const topic_name = nh.param<std::string>("image_topic", "/camera/image_raw");
    auto const pc_name = nh.param<std::string>("pc_topic", "/camera/points");
    auto const depth = nh.param<double>("depth", 1.0);
    auto const width = nh.param<double>("width", 1.6);
    auto const height = nh.param<double>("height", 0.9);
    auto const frame_id = nh.param<std::string>("frame_id", "");
    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>(pc_name, 1);
    ros::Subscriber im_sub = nh.subscribe<sensor_msgs::Image>(topic_name, 1, [&](sensor_msgs::ImageConstPtr msg) {
        auto img = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        sensor_msgs::PointCloud2Ptr pc = sunshine::getFlatPointCloud(img->image, width, height, depth, msg->header);
        if (!frame_id.empty()) {
            pc->header.frame_id = frame_id;
        }
        pc_pub.publish(pc);
    });

    ros::spin();
}
