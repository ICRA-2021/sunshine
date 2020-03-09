//
// Created by stewart on 3/4/20.
//

#ifndef SUNSHINE_PROJECT_ROS_UTILS_HPP
#define SUNSHINE_PROJECT_ROS_UTILS_HPP

#include <sensor_msgs/PointCloud2.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include "colors.hpp"

namespace sunshine {
geometry_msgs::TransformStamped broadcastTranform(std::string frame_id,
                                                  tf2::Vector3 origin,
                                                  tf2::Quaternion q,
                                                  std::string parentFrame = "map",
                                                  ros::Time stamp = ros::Time::now()) {
    static tf2_ros::TransformBroadcaster br;
    tf2::Transform transform;
    transform.setOrigin(origin);
    transform.setRotation(q);
    auto stampedTransform = tf2::toMsg(tf2::Stamped<tf2::Transform>(transform, stamp, parentFrame));
    stampedTransform.child_frame_id = frame_id;
    br.sendTransform(stampedTransform);
    return stampedTransform;
}

template<typename T>
inline void transformPose(geometry_msgs::Point &out,
                          std::vector<T> const &poses,
                          size_t const &idx,
                          geometry_msgs::TransformStamped const &transform) {
    geometry_msgs::Point point;
    point.x = poses[idx];
    point.y = poses[idx + 1];
    point.z = poses[idx + 2];
    tf2::doTransform(point, out, transform);
}

/**
 * @brief createPointCloud
 * @param width
 * @param height
 * @param colorFieldName
 * @return
 */
sensor_msgs::PointCloud2Ptr createPointCloud(uint32_t width,
                                             uint32_t height,
                                             std::string colorFieldName = "rgba",
                                             std_msgs::Header const &header = {}) {
    sensor_msgs::PointCloud2Ptr pc(new sensor_msgs::PointCloud2());
    sensor_msgs::PointField basePointField;
    basePointField.datatype = basePointField.FLOAT32;
    basePointField.count = 1;

    auto offset = 0u;
    for (auto const &field : {"x", "y", "z"}) {
        sensor_msgs::PointField pointField(basePointField);
        pointField.name = field;
        pointField.offset = offset;
        pc->fields.push_back(pointField);
        pc->point_step += sizeof(float);
        offset += sizeof(float);
    }

    if (!colorFieldName.empty()) {
        sensor_msgs::PointField colorField;
        colorField.datatype = colorField.UINT32;
        colorField.count = 1;
        colorField.offset = offset;
        colorField.name = colorFieldName;
        pc->fields.push_back(colorField);
        pc->point_step += sizeof(uint32_t);
    }

    pc->width = width;
    pc->height = height;
    pc->is_dense = true;
    pc->row_step = pc->point_step * pc->width;
    pc->data = std::vector<uint8_t>(pc->row_step * pc->height);
    pc->header = header;
    return pc;
}

struct RGBAPoint {
  double x, y, z;
  RGBA color;
};

union RGBAPointCloudElement {
  uint8_t bytes[sizeof(float) * 3 + sizeof(uint32_t)]; // to enforce size
  struct {
    float x;
    float y;
    float z;
    std::array<uint8_t, 4> rgba;
  } data;
};

template<typename PointContainer>
sensor_msgs::PointCloud2Ptr toRGBAPointCloud(PointContainer const &points, std::string frame_id = "map") {
    auto const count = points.size();
    uint32_t const height = 1, width = uint32_t(count);
    sensor_msgs::PointCloud2Ptr pc = createPointCloud(width, height, "rgba");

    if (pc->point_step != sizeof(RGBAPointCloudElement)) {
        throw std::invalid_argument("Point cloud point_step suggests fields are not xyz[rgba]");
    }

    pc->header.frame_id = frame_id;
    RGBAPointCloudElement *pcIterator = reinterpret_cast<RGBAPointCloudElement *>(pc->data.data());

    for (decltype(points.size()) i = 0; i < count; i++) {
        RGBAPoint const &point = points[i];
        pcIterator->data.x = static_cast<float>(point.x);
        pcIterator->data.y = static_cast<float>(point.y);
        pcIterator->data.z = static_cast<float>(point.z);
        pcIterator->data.rgba = static_cast<std::array<uint8_t, 4>>(point.color);
        pcIterator++;
    }
    return pc;
}
}

#endif //SUNSHINE_PROJECT_ROS_UTILS_HPP
