// From http://wiki.ros.org/image_transport/Tutorials/PublishingImages

#include "image_utils.hpp"
#include <cv_bridge/cv_bridge.h>
//#if __has_include(<ds_sensor_msgs/Dvl.h>)
#include <ds_sensor_msgs/Dvl.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <cmath>
//#endif
#include <ros/ros.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pc_publisher");
    ros::NodeHandle nh("~");
    auto const topic_name = nh.param<std::string>("image_topic", "/camera/image_raw");
    auto const pc_name = nh.param<std::string>("pc_topic", "/camera/points");
    auto const altitudeOffset = nh.param<double>("altitude", 1.0);
    auto const preset_width = nh.param<double>("width", 1.5);
    auto const preset_height = nh.param<double>("height", 1.0);
    auto const focal_distance = nh.param<double>("focal_distance_mm", 35) / 1000.;
    auto const fovX = nh.param<double>("fov_x", 54.4) * M_PI / 180.;
    auto const fovY = nh.param<double>("fov_y", 37.9) * M_PI / 180.;
    auto const altRef = nh.param<std::string>("altitude_ref", "");
    auto const frame_id = nh.param<std::string>("frame_id", "");
    ros::Subscriber altitude_sub;
    double altitude = altitudeOffset;
    bool altRefReceived = false;
//#if __has_include(<ds_sensor_msgs/Dvl.h>)
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener(tf_buffer);
    if (!altRef.empty()) {
        altitude_sub = nh.subscribe<ds_sensor_msgs::Dvl>(altRef, 1, [&](ds_sensor_msgs::DvlConstPtr msg) {
            //            assert(msg->coordinate_mode == ds_sensor_msgs::Dvl::DVL_COORD_INSTRUMENT);
            auto const& beam_vecs = msg->beam_unit_vec;
            auto const& beam_ranges = msg->range;
            try {
                auto const transform_msg = tf_buffer.lookupTransform(frame_id, msg->header.frame_id, ros::Time(), ros::Duration(1));
                tf2::Vector3 seafloor_vec;
                for (auto i = 0u; i < beam_vecs.size(); i++) {
                    tf2::Vector3 beam_vec(beam_vecs[i].x, beam_vecs[i].y, beam_vecs[i].z);
                    seafloor_vec += (beam_vec * beam_ranges[i]);
                }
                seafloor_vec /= beam_vecs.size();
                tf2::Transform transform;
                tf2::fromMsg(transform_msg.transform, transform);
                tf2::Vector3 const seafloor_vec_ref_frame = transform.getBasis() * seafloor_vec;
                altitude = seafloor_vec_ref_frame.getZ() + altitudeOffset;
                ROS_INFO("Computed altitude from DVL: %f", altitude);
                altRefReceived = true;
            } catch (tf2::LookupException) {
                ROS_ERROR("Failed to find transform from DVL frame!");
                return;
            }
        });
    }
//#endif
    auto width = preset_width;
    auto height = preset_height;
    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>(pc_name, 1);
    ros::Subscriber im_sub = nh.subscribe<sensor_msgs::Image>(topic_name, 1, [&](sensor_msgs::ImageConstPtr msg) {
        if (!altRef.empty()) {
            assert(fovX > 0 && fovY > 0);
            if (!altRefReceived) {
                ROS_ERROR("Ignoring image! Altitude reference has not been received...");
                return;
            }
            width = 2. * std::abs(altitude) * std::tan(fovX / (2.));
            height = 2. * std::abs(altitude) * std::tan(fovY / (2.));
            ROS_INFO("Computed image width %f and height %f from altitude %f", width, height, altitude);
        }
        auto img = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        sensor_msgs::PointCloud2Ptr pc = sunshine::getFlatPointCloud(img->image, width, height, altitude, msg->header);
        if (!frame_id.empty()) {
            pc->header.frame_id = frame_id;
        }
        pc_pub.publish(pc);
    });

    ros::spin();
}
