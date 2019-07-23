// From http://wiki.ros.org/image_transport/Tutorials/PublishingImages

#include "image_utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_listener.h>
//#if __has_include(<ds_sensor_msgs/Dvl.h>)
#include <cmath>
#include <ds_sensor_msgs/Dvl.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
//#endif
#include <ros/ros.h>
#include <numeric>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pc_publisher");
    ros::NodeHandle nh("~");
    auto const topic_name = nh.param<std::string>("image_topic", "/camera/image_raw");
    auto const pc_name = nh.param<std::string>("pc_topic", "/camera/points");
    auto const fixed_altitude = nh.param<double>("altitude", 1.0);
    auto const preset_width = nh.param<double>("width", 1.5);
    auto const preset_height = nh.param<double>("height", 1.0);
    auto const zero_z = nh.param<bool>("zero_z", false);
    auto const fovX = nh.param<double>("fov_x", 54.4) * M_PI / 180.;
    auto const fovY = nh.param<double>("fov_y", 37.9) * M_PI / 180.;
    auto const altRef = nh.param<std::string>("altitude_ref", "");
    auto const frame_id = nh.param<std::string>("frame_id", "");
    auto const map_frame_id = nh.param<std::string>("map_frame_id", "");
    auto const zeroz_frame_id = nh.param<std::string>("zeroz_frame_id", "zeroz_sensor_frame");
    ros::Subscriber altitude_sub;
    double altitude = fixed_altitude;
    bool altRefReceived = false;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener(tf_buffer);
    tf2_ros::TransformBroadcaster tf_broadcaster;
    //#if __has_include(<ds_sensor_msgs/Dvl.h>)
    if (!altRef.empty()) {
        altitude_sub = nh.subscribe<ds_sensor_msgs::Dvl>(altRef, 1, [&](ds_sensor_msgs::DvlConstPtr msg) {
            auto const& beam_vecs = msg->beam_unit_vec;
            auto const& beam_ranges = msg->range;
            tf2::Vector3 seafloor_vec;
            std::vector<double> z_estimates(beam_vecs.size(), 0.0);
            for (auto i = 0u; i < beam_vecs.size(); i++) {
                tf2::Vector3 beam_vec(beam_vecs[i].x, beam_vecs[i].y, beam_vecs[i].z);
                z_estimates[i] += (beam_vec * beam_ranges[i]).getZ();
            }

            double const alt_estimate = std::accumulate(z_estimates.begin(), z_estimates.end(), 0.) / beam_vecs.size();
            double const outlier = *std::max_element(z_estimates.begin(), z_estimates.end(), [=](double const& a, double const& b){
                return std::abs(a - alt_estimate) < std::abs(b - alt_estimate);
            });

            altitude = ((alt_estimate * beam_vecs.size()) - outlier) / (beam_vecs.size() - 1);
            ROS_DEBUG("Computed altitude from DVL: %f (removed outlier %f)", altitude, outlier);
            altRefReceived = true;
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
        sensor_msgs::PointCloud2Ptr pc;
        if (zero_z) {
            assert(!map_frame_id.empty());
            try {
                auto const vehicleTransform = tf_buffer.lookupTransform(map_frame_id, (frame_id.empty()) ? msg->header.frame_id : frame_id, ros::Time(msg->header.stamp), ros::Duration(1));
                auto fakeSensorTransform = vehicleTransform;
                fakeSensorTransform.child_frame_id = zeroz_frame_id;
                fakeSensorTransform.transform.translation.z = 0.0;
                fakeSensorTransform.header.stamp = msg->header.stamp;
                tf_broadcaster.sendTransform(fakeSensorTransform);
                pc = sunshine::getFlatPointCloud(img->image, width, height, 0.0, msg->header);
                pc->header.frame_id = zeroz_frame_id;
            } catch (tf2::ExtrapolationException) {
                ROS_ERROR("Failed to find transform from %s to %s!", map_frame_id.c_str(), (frame_id.empty()) ? msg->header.frame_id.c_str() : frame_id.c_str());
                return;
            }
        } else {
            pc = sunshine::getFlatPointCloud(img->image, width, height, altitude, msg->header);
            if (!frame_id.empty()) {
                pc->header.frame_id = frame_id;
            }
        }
        pc_pub.publish(pc);
    });

    ros::spin();
}
