#include "image_utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using namespace sunshine;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_walker");
    std::string const image_filename = (argc > 1) ? argv[1] : "";

    ros::NodeHandle nh("~");
    auto const fps = nh.param<double>("fps", 30);
    auto const speed = nh.param<double>("speed", 1);
    auto const altitude = nh.param<double>("height", -1);
    auto const overlap = nh.param<int>("overlap", 0);
    auto const col_major = nh.param<bool>("col_major", false);
    auto const size = nh.param<std::string>("size", "0x0");
    auto const scale = nh.param<double>("scale", 1);
    auto const image_name = nh.param<std::string>("image", image_filename);
    auto const image_topic = nh.param<std::string>("image_topic", "~/image");
    auto const depth_cloud_topic = nh.param<std::string>("depth_cloud_topic", "~/cloud");
    auto const depth_image_topic = nh.param<std::string>("depth_image_topic", "~/depth");
    auto const transform_topic = nh.param<std::string>("transform_topic", "/tf");
    auto const frame_id = nh.param<std::string>("frame_id", "base_link");
    auto const pixel_scale = nh.param<double>("pixel_scale", 0.01);

    cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
    cv::waitKey(30);

    if (fps <= 0) {
        throw std::invalid_argument("fps must be positive!");
    }

    if (size.find('x') == std::string::npos) {
        throw std::invalid_argument("Size must be in the format of LxW");
    }

    auto width = std::stoi(size.substr(0, size.find('x')));
    auto height = std::stoi(size.substr(size.find('x') + 1));
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Size (both L and W) must be set.");
    }
    if (overlap < 0) {
        throw std::invalid_argument("Overlap cannot be negative!");
    } else if ((col_major && overlap >= width) || (!col_major && overlap >= height)) {
        throw std::invalid_argument("Overlap is too large!");
    }


    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = frame_id;

    // Publish image topic
    image_transport::ImageTransport it(nh);
    image_transport::Publisher imagePub = it.advertise(image_topic, 1);

    // Publish depth image topic
    bool const publishDepthImage = (altitude > 0);
    sensor_msgs::PointCloud2Ptr depthCloud;
    ros::Publisher depthCloudPub;
    image_transport::Publisher depthImagePub;
    std::unique_ptr<sunshine::ImageScanner> image_scanner;
    if (publishDepthImage) {
        ROS_INFO("Publishing depth image.");
        depthCloudPub = nh.advertise<sensor_msgs::PointCloud2>(depth_cloud_topic, 1);
        depthImagePub = it.advertise(depth_image_topic, 1);
        image_scanner = std::make_unique<sunshine::ImageScanner3D<>>(image, width, height,
                                                                     getFlatHeightMap(image.cols, image.rows, 0.),
                                                                     scale, altitude, pixel_scale);
    } else {
        image_scanner = std::make_unique<sunshine::ImageScanner>(image, width, height, scale);
    }

    double cx = image_scanner->getMinX();
    double cy = image_scanner->getMinY();
    double track = (col_major) ? cx : cy;

    auto const poseCallback = [&]() {
        tf2::Quaternion q;
        q.setRPY(0, M_PI, M_PI);
        broadcastTranform(frame_id,
                          tf2::Vector3(image_scanner->getX(), -image_scanner->getY(), (altitude > 0) ? altitude : 0), q,
                          "map", header.stamp);
    };
    if (publishDepthImage) {
        poseCallback();
        ros::spinOnce();
    }

    enum class Direction {
        DOWN = -2, LEFT = -1, RIGHT = 1, UP = 2, NONE = 0
    };
    Direction dir = Direction::RIGHT;

    auto &primary_axis = (col_major) ? cy : cx;
    auto const primary_axis_max = ((col_major) ? image_scanner->getMaxY() : image_scanner->getMaxX());
    auto const primary_window_extent = (col_major) ? image_scanner->getMinY() : image_scanner->getMinX();
    auto &secondary_axis = (col_major) ? cx : cy;
    auto const secondary_axis_max = ((col_major) ? image_scanner->getMaxX() : image_scanner->getMaxY());
    auto const secondary_window_extent = (col_major) ? image_scanner->getMinX() : image_scanner->getMinY();
    auto const step_size = speed / fps;

    auto const advanceTrack = [&]() {
        if (track >= secondary_axis_max - secondary_window_extent) {
            dir = Direction::NONE;
            return;
        }
        dir = Direction::DOWN;
        track += std::min(2 * secondary_window_extent - overlap, secondary_axis_max - secondary_window_extent - track);
    };

    auto const lawnmowerMove = [&](double const step_size) {
        if (dir == Direction::RIGHT) {
            if (primary_axis >= primary_axis_max - primary_window_extent) {
                advanceTrack();
            } else {
                primary_axis += std::min(step_size, primary_axis_max - primary_window_extent - primary_axis);
            }
        } else if (dir == Direction::LEFT) {
            if (primary_axis <= primary_window_extent) {
                advanceTrack();
            } else {
                primary_axis -= std::min(step_size, primary_axis - primary_window_extent);
            }
        }

        if (dir == Direction::DOWN) {
            // Advance towards the new track
            secondary_axis += std::min(step_size, track - secondary_axis);
            if (secondary_axis >= track) {
                dir = (primary_axis <= primary_window_extent) ? Direction::RIGHT : Direction::LEFT;
            }
        }
    };

    ros::Rate rate(fps);
    uint64_t warmup = 8;
    for (uint64_t numFrames = 0; nh.ok(); numFrames++) {
        header.stamp = ros::Time::now();
        image_scanner->moveTo(cx, cy);
        auto const &visibleRegion = image_scanner->getCurrentView();
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", visibleRegion).toImageMsg();
        if (publishDepthImage) {
            poseCallback();
            auto *image_scanner_3d = dynamic_cast<sunshine::ImageScanner3D<> *>(image_scanner.get());
            depthCloud = image_scanner_3d->getCurrentPointCloud();
            depthCloud->header = header;
            depthCloudPub.publish(*depthCloud);
            depthImagePub.publish(cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1,
                                                     image_scanner_3d->getCurrentDepthView()).toImageMsg());
        }
        imagePub.publish(msg);

        ros::spinOnce();
        rate.sleep();

        if (numFrames > warmup) {
            lawnmowerMove(step_size);
        }
    }
}
