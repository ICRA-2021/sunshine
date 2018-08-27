#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

template <typename ValType>
struct cvType {
};
template <>
struct cvType<float> {
    static int const value = CV_32F;
};
template <>
struct cvType<double> {
    static int const value = CV_64F;
};
template <>
struct cvType<uint8_t> {
    static int const value = CV_8U;
};
template <>
struct cvType<uint16_t> {
    static int const value = CV_16U;
};

/**
 * @brief toMat Converts two parallel vectors, of 2D poses (xy-coordinates) and 1D values, respectively, to a cv::Mat object.
 * @param idxes 2D poses of the form [x1,y1,...,xN,yN] (size of 2*N)
 * @param values Scalar values corresponding to each pose of the form [z1,...,zN] (size of N)
 * @return cv::Mat where mat.at<MatValueType>(yI, xI) = zI
 */
template <typename IdxType, typename ValueType, typename MatValueType = ValueType>
cv::Mat toMat(std::vector<IdxType> const& idxes, std::vector<ValueType> const& values)
{
    assert(idxes.size() == values.size() * 2);

    // Compute the required size of the matrix based on the largest (x,y) coordinate
    IdxType max_x = 0, max_y = 0;
    for (auto i = 0ul; i < values.size(); i++) {
        assert(idxes[i * 2] >= 0 && idxes[i * 2 + 1] >= 0);
        max_x = std::max(max_x, idxes[i * 2]);
        max_y = std::max(max_y, idxes[i * 2 + 1]);
    }
    cv::Mat mat(max_y + 1, max_x + 1, cvType<MatValueType>::value); // Don't forget to add 1 to each dimension (because 0-indexed)

    for (auto i = 0ul; i < values.size(); i++) {
        mat.at<MatValueType>(idxes[i * 2 + 1], idxes[i * 2]) = values[i]; // mat.at(y,x) = z
    }
    return mat;
}

geometry_msgs::TransformStamped broadcastTranform(std::string frame_id, tf2::Vector3 origin, tf2::Quaternion q, std::string parentFrame = "map", ros::Time stamp = ros::Time::now())
{
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
inline void transformPose(geometry_msgs::Point& out, std::vector<T> const& poses, size_t const& idx, geometry_msgs::TransformStamped const& transform) {
    geometry_msgs::Point point;
    point.x = poses[idx];
    point.y = poses[idx+1];
    point.z = poses[idx+2];
    tf2::doTransform(point, out, transform);
}

#endif // UTILS_HPP
