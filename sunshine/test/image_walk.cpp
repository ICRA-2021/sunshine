#include "utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

typedef float_t DepthType;

cv::Mat getVisibleRegion(cv::Mat const& image, float cx, float cy, int width, int height)
{
    cv::Rect roi;
    roi.x = std::round(cx - width / 2.);
    roi.y = std::round(cy - height / 2.);
    roi.width = width;
    roi.height = height;
    return image(roi);
}

cv::Mat scaleImage(cv::Mat const& image, float scale)
{
    cv::Mat scaled(std::ceil(image.rows * scale), std::ceil(image.cols * scale), image.type());
    cv::resize(image, scaled, scaled.size(), 0, 0, cv::INTER_CUBIC);
    return scaled;
}

void copyImageColorToPointCloud(cv::Mat const& image, sensor_msgs::PointCloud2& pointCloud)
{
    auto const& colorField = std::find_if(pointCloud.fields.begin(), pointCloud.fields.end(), [](decltype(pointCloud.fields)::value_type const& field) {
        return field.name == "rgb";
    });
    if (colorField == pointCloud.fields.end()) {
        throw std::invalid_argument("Cannot find color field in point cloud.");
    }
    assert(static_cast<uint64_t>(image.rows * image.cols) == pointCloud.width * pointCloud.height);
    uint8_t* dataOffset = pointCloud.data.data() + colorField->offset;
    size_t dataStep = pointCloud.point_step;
    for (auto row = 0; row < image.rows; row++) {
        for (auto col = 0; col < image.cols; col++) {
            auto const color = image.at<cv::Vec3b>(row, col);
            for (auto i = 0u; i < 3; i++)
                *(dataOffset + i) = color.val[i];
            dataOffset += dataStep;
        }
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_publisher");

    ros::NodeHandle nh("~");
    auto const fps = nh.param<float>("fps", 30);
    auto const speed = nh.param<float>("speed", 30);
    auto const altitude = nh.param<float>("height", -1);
    auto const fov = nh.param<float>("fov", -1);
    auto const overlap = nh.param<int>("overlap", 0);
    auto const col_major = nh.param<bool>("col_major", false);
    auto const size = nh.param<std::string>("size", "0x0");
    auto const scale = nh.param<float>("scale", 1);
    auto const image_name = nh.param<std::string>("image", argv[1]);
    auto const image_topic = nh.param<std::string>("image_topic", "/camera/image");
    auto const depth_cloud_topic = nh.param<std::string>("depth_cloud_topic", "/camera/test_point_cloud");
    auto const depth_image_topic = nh.param<std::string>("depth_image_topic", "/camera/depth");
    auto const pixel_scale = nh.param<double>("pixel_scale", 0.01);

    cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
    cv::waitKey(30);

    if (fps <= 0) {
        throw std::invalid_argument("fps cannot be non-positive!");
    }

    if (size.find("x") == size.npos) {
        throw std::invalid_argument("Size must be in the format of LxW");
    }

    auto width = std::stoi(size.substr(0, size.find("x")));
    auto height = std::stoi(size.substr(size.find("x") + 1));
    if (width <= 0 || height <= 0) {
        if (fov <= 0 || altitude <= 0) {
            throw std::invalid_argument("Either size (both L and W) must be set or both FOV and height must be set.");
        }

        throw std::runtime_error("FOV is not implemented."); // TODO: FOV calculation
    } else if (fov > 0 && altitude > 0) {
        throw std::invalid_argument("Cannot set all of size, FOV, and height! Choose at most two.");
    }

    if (overlap < 0) {
        throw std::invalid_argument("Overlap cannot be negative!");
    } else if ((col_major && overlap >= width) || (!col_major && overlap >= height)) {
        throw std::invalid_argument("Overlap is too large!");
    }

    float x = width / 2.f;
    float y = height / 2.f;
    float track = (col_major) ? x : y;

    auto const poseCallback = [&](const ros::TimerEvent&) {
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(x, -y, (altitude >= 0) ? altitude : 0) * pixel_scale);
        tf::Quaternion q;
        q.setRPY(0, M_PI, M_PI);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "camera"));
    };

    // Publish image topic
    image_transport::ImageTransport it(nh);
    image_transport::Publisher imagePub = it.advertise(image_topic, 1);

    // Publish depth image topic
    bool const publishDepthImage = (fov > 0 || altitude > 0);
    cv::Mat depthImage(height, width, cvType<DepthType>::value);
    sensor_msgs::PointCloud2 depthCloud;
    ros::Publisher depthCloudPub;
    image_transport::Publisher depthImagePub;
    if (publishDepthImage) {
        ROS_INFO("Publishing depth image.");
        sensor_msgs::PointField basePointField;
        basePointField.datatype = basePointField.FLOAT32;
        basePointField.count = 1;

        auto offset = 0u;
        for (auto const& field : { "x", "y", "z" }) {
            sensor_msgs::PointField pointField = basePointField;
            pointField.name = field;
            pointField.offset = offset;
            depthCloud.fields.push_back(pointField);
            offset += sizeof(float);
        }

        sensor_msgs::PointField colorField;
        colorField.datatype = colorField.UINT32;
        colorField.count = 1;
        colorField.offset = offset;
        colorField.name = "rgb";
        depthCloud.fields.push_back(colorField);

        depthCloud.header.frame_id = "camera";
        depthCloud.width = uint32_t(width);
        depthCloud.height = uint32_t(height);
        depthCloud.is_dense = true;
        depthCloud.point_step = sizeof(float) * 3 + sizeof(uint32_t);
        depthCloud.row_step = depthCloud.point_step * uint32_t(width);
        depthCloud.data = std::vector<uint8_t>(depthCloud.row_step * size_t(height));
        float* depthCloudIterator = reinterpret_cast<float*>(&(depthCloud.data.data()[0]));

        depthCloudPub = nh.advertise<sensor_msgs::PointCloud2>(depth_cloud_topic, 1);
        depthImagePub = it.advertise(depth_image_topic, 1);
        for (auto row = 0; row < height; row++) {
            for (auto col = 0; col < width; col++) {
                double const depth = std::pow(std::pow(row - y, 2) + std::pow(col - x, 2) + std::pow(altitude, 2), 1. / 2.) * pixel_scale;
                depthImage.at<DepthType>(row, col) = static_cast<DepthType>(depth);
                *(depthCloudIterator++) = (col - width / 2.f) * pixel_scale;
                *(depthCloudIterator++) = (row - height / 2.f) * pixel_scale;
                *(depthCloudIterator++) = altitude * pixel_scale;
                static_assert(sizeof(float) == sizeof(uint32_t), "Following code assumes that sizeof(float) == sizeof(uint32_t)!");
                depthCloudIterator++; // skip color
            }
        }
    }

    enum class Direction {
        DOWN = -2,
        LEFT = -1,
        RIGHT = 1,
        UP = 2,
        NONE = 0
    };
    Direction dir = Direction::RIGHT;

    auto& primary_axis = (col_major) ? y : x;
    auto const primary_axis_max = ((col_major) ? image.rows : image.cols);
    auto const primary_window_extent = ((col_major) ? height : width) / 2.f;
    auto& secondary_axis = (col_major) ? x : y;
    auto const secondary_axis_max = ((col_major) ? image.cols : image.rows);
    auto const secondary_window_extent = ((col_major) ? width : height) / 2.f;
    auto const step_size = speed / fps;

    auto const advanceTrack = [&]() {
        if (track >= secondary_axis_max - secondary_window_extent) {
            dir = Direction::NONE;
            return;
        }
        dir = Direction::DOWN;
        track += std::min(2 * secondary_window_extent - overlap, secondary_axis_max - secondary_window_extent - track);
    };

    auto const lawnmowerMove = [&]() {
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
    auto poseTimer = nh.createTimer(rate.cycleTime(), poseCallback, false, true);
    std_msgs::Header header;
    header.frame_id = "camera";
    while (nh.ok()) {
        auto const& visibleRegion = getVisibleRegion(image, x, y, width, height);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", scaleImage(visibleRegion, scale)).toImageMsg();
        imagePub.publish(msg);
        if (publishDepthImage) {
            copyImageColorToPointCloud(visibleRegion, depthCloud);
            depthCloudPub.publish(depthCloud);
            depthImagePub.publish(cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, depthImage).toImageMsg());
        }

        ros::spinOnce();
        lawnmowerMove();
        rate.sleep();
    }
}
