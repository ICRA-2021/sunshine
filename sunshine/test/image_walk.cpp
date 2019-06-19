#include "image_utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

using namespace sunshine;

class MovePattern {
public:
    virtual ~MovePattern() = default;
    virtual void move() = 0;
    virtual double getX() const = 0;
    virtual double getY() const = 0;
};

class BoustrophedonicPattern : public MovePattern {
public:
    enum class Direction {
        DOWN = -2,
        LEFT = -1,
        RIGHT = 1,
        UP = 2,
        NONE = 0
    };

private:
    sunshine::ImageScanner* image_scanner;
    bool col_major;
    double x;
    double y;
    double const step_size;
    double const overlap;
    Direction dir;

    double &primary_axis = (col_major) ? y : x, secondary_axis = (col_major) ? x : y;
    double& track = (col_major) ? x : y;
    double const primary_axis_max = ((col_major) ? image_scanner->getMaxY() : image_scanner->getMaxX());
    double const primary_window_extent = (col_major) ? image_scanner->getMinY() : image_scanner->getMinX();
    double const secondary_axis_max = ((col_major) ? image_scanner->getMaxX() : image_scanner->getMaxY());
    double const secondary_window_extent = (col_major) ? image_scanner->getMinX() : image_scanner->getMinY();

    void advance_track()
    {
        if (track >= secondary_axis_max - secondary_window_extent) {
            dir = Direction::NONE;
            return;
        }
        dir = Direction::DOWN;
        track += std::min(2 * secondary_window_extent - overlap, secondary_axis_max - secondary_window_extent - track);
    }

public:
    BoustrophedonicPattern(sunshine::ImageScanner* image_scanner, double initial_x, double initial_y, bool col_major, double step_size, double overlap, Direction dir = Direction::RIGHT)
        : image_scanner(image_scanner)
        , col_major(col_major)
        , x(initial_x)
        , y(initial_y)
        , step_size(step_size)
        , overlap(overlap)
        , dir(dir)
    {
    }

    ~BoustrophedonicPattern() = default;

    void move()
    {
        if (dir == Direction::RIGHT) {
            if (primary_axis >= primary_axis_max - primary_window_extent) {
                advance_track();
            } else {
                primary_axis += std::min(step_size, primary_axis_max - primary_window_extent - primary_axis);
            }
        } else if (dir == Direction::LEFT) {
            if (primary_axis <= primary_window_extent) {
                advance_track();
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
    }

    double getX() const
    {
        return x;
    }
    double getY() const
    {
        return y;
    }
};

int main(int argc, char** argv)
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
    auto const pattern_name = nh.param<std::string>("move_pattern", "lawnmower");
    bool const follow_mode = pattern_name.empty() || pattern_name == "follow";
    auto const follow_topic = nh.param<std::string>("follow_topic", "");

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

    ros::Subscriber follow_sub;
    if (!follow_topic.empty()) {
        follow_sub = nh.subscribe<sensor_msgs::Image>(follow_topic, 1, [&](sensor_msgs::ImageConstPtr img) {
            if (frame_id.empty()) {
                header = img->header;
            } else {
                header.stamp = img->header.stamp;
                header.seq = img->header.seq;
            } });
    }

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

    auto const poseCallback = [&]() {
        tf2::Quaternion q;
        q.setRPY(0, M_PI, M_PI);
        broadcastTranform(frame_id,
            tf2::Vector3(image_scanner->getX(), -image_scanner->getY(), (altitude > 0) ? altitude : 0), q,
            "map", header.stamp);
    };
    if (!follow_mode) {
        poseCallback();
        ros::spinOnce();
    }

    std::unique_ptr<MovePattern> movePattern;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener;
    if (follow_mode) {
        tf_buffer = std::make_unique<tf2_ros::Buffer>();
        tf_listener = std::make_unique<tf2_ros::TransformListener>(*tf_buffer);
    } else if (pattern_name == "lawnmower") {
        movePattern = std::make_unique<BoustrophedonicPattern>(image_scanner.get(), cx, cy, col_major, speed / fps, overlap);
    }

    ros::Rate rate(fps);
    uint64_t const warmup = 8;
    for (uint64_t numFrames = 0; nh.ok(); numFrames++) {
        // Update robot pose
        if (follow_mode) {
            try {
                geometry_msgs::TransformStamped transform_msg;
                if (follow_topic.empty()) {
                    transform_msg = tf_buffer->lookupTransform("map", frame_id, ros::Time(0));
                    header.stamp = transform_msg.header.stamp;
                } else {
                    transform_msg = tf_buffer->lookupTransform("map", header.frame_id, ros::Time(header.stamp));
                }
                cx = transform_msg.transform.translation.x;
                cy = -transform_msg.transform.translation.y;
            } catch (tf2::LookupException ex) {
                ROS_WARN_THROTTLE(1, "Failed to find tranform from %s to %s", frame_id.c_str(), "map");
            }
        } else {
            static_assert(warmup >= 1, "Warmup should be at least 1 or initial frame will be missing.");
            if (numFrames > warmup) {
                movePattern->move();
            }
            cx = movePattern->getX();
            cy = movePattern->getY();

            // publish update pose (since we control it)
            poseCallback();
            ros::spinOnce();

            // update header timestamp for published images
            header.stamp = ros::Time::now();
        }

        // Publish images
        image_scanner->moveTo(cx, cy);
        auto const& visibleRegion = image_scanner->getCurrentView();
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", visibleRegion).toImageMsg();
        if (publishDepthImage) {
            auto* image_scanner_3d = dynamic_cast<sunshine::ImageScanner3D<>*>(image_scanner.get());
            depthCloud = image_scanner_3d->getCurrentPointCloud();
            depthCloud->header = header;
            depthCloudPub.publish(*depthCloud);
            depthImagePub.publish(cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1,
                image_scanner_3d->getCurrentDepthView())
                                      .toImageMsg());
        }
        imagePub.publish(msg);

        ros::spinOnce();
        rate.sleep();
    }
}
