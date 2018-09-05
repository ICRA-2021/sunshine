// From http://wiki.ros.org/image_transport/Tutorials/PublishingImages

#include "utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_publisher");

    ros::NodeHandle nh("~");
    auto const loop_rate = nh.param<double>("loop_rate", 30);
    auto const topic_name = nh.param<std::string>("image_topic", "/camera/image");
    auto const transform_topic = nh.param<std::string>("transform_topic", "/camera_world_transform");
    auto const frame_id = nh.param<std::string>("frame_id", "camera");
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise(topic_name, 1);
    ros::Publisher tfPublisher = nh.advertise<geometry_msgs::TransformStamped>(transform_topic, 1);
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::waitKey(30);
    std_msgs::Header header;
    header.frame_id = frame_id;
    tf2::Quaternion q;
    q.setRPY(0, M_PI, M_PI);

    if (loop_rate <= 0) {
        throw std::invalid_argument("loop_rate (Hz) cannot be non-positive!");
    }

    ros::Rate rate(loop_rate);
    while (nh.ok()) {
        header.stamp = ros::Time::now();
        tfPublisher.publish(sunshine::broadcastTranform(frame_id, {0,0,0}, q, "map", header.stamp));
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        pub.publish(msg);
            ros::spinOnce();
        rate.sleep();
    }
}
