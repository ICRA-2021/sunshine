//
// Created by stewart on 2020-07-06.
//

#ifndef SUNSHINE_PROJECT_VISUAL_WORD_EXTRACTOR_HPP
#define SUNSHINE_PROJECT_VISUAL_WORD_EXTRACTOR_HPP

#include "sunshine/depth_adapter.hpp"
#include "sunshine/common/ros_conversions.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>

#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "ros/ros.h"
#include "sensor_msgs/image_encodings.h"

#include "sensor_msgs/PointCloud2.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include <boost/filesystem.hpp>

#include <rosbag/bag.h>
#include <rosbag/view.h>

using namespace std;

namespace sunshine {

template <class WordAdapterType>
class WordExtractorNode {
    ros::Publisher words_pub, words_2d_pub;
    geometry_msgs::TransformStamped latest_transform;
    bool transform_recvd; // TODO: smarter way of handling stale/missing poses
    bool pc_recvd;
    bool use_pc;
    bool use_tf;
    bool publish_2d;
    bool publish_3d;
    double rate;
    std::string frame_id = "";
    std::string world_frame_name = "";
    std::string sensor_frame_name = "";
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;

    WordAdapterType wordAdapter;
    WordDepthAdapter depthAdapter;

    image_transport::ImageTransport it;
    image_transport::Subscriber imageSub;
    ros::Subscriber transformSub;
    ros::Subscriber depthSub;

    void transformCallback(const geometry_msgs::TransformStampedConstPtr &msg) {
        // Callback to handle world to sensor transform
        if (!transform_recvd) {
            ROS_WARN("Using a dedicated transform topic is inferior to looking up tf2 transforms!");
            transform_recvd = true;
        }

        latest_transform = *msg;
        frame_id = latest_transform.header.frame_id;
    }

    void pcCallback(const sensor_msgs::PointCloud2ConstPtr &msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pc);

        depthAdapter.updatePointCloud(pc);
        pc_recvd = true;
    }

    void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
        // TODO: Add logic to handle compressed images. Refer to examples from Nathan/Vv
        cv_bridge::CvImageConstPtr img_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);

        auto const imgObs = std::make_unique<ImageObservation>(sensor_frame_name, msg->header.stamp.toSec(), msg->header.seq, img_ptr->image);
        auto const wordObs = wordAdapter(imgObs.get());
        if (publish_2d) words_2d_pub.publish(toRosMsg(*wordObs));

        if (publish_3d) {
            if (use_pc && !pc_recvd) {
                ROS_ERROR("No point cloud received, observations will not be published");
                return;
            }

            auto wordObs3d = depthAdapter(wordObs.get());
            geometry_msgs::TransformStamped observation_transform;

            if (use_tf) {
                if (transform_recvd) {
                    assert(frame_id == sensor_frame_name);
                    observation_transform = latest_transform;
                } else {
                    try {
                        geometry_msgs::TransformStamped transform_msg;
                        transform_msg = tf_buffer.lookupTransform(world_frame_name,
                                                                  sensor_frame_name,
                                                                  msg->header.stamp,
                                                                  ros::Duration(0.5 / rate));
                        observation_transform = transform_msg;
                    } catch (tf2::TransformException const &ex) {
                        ROS_ERROR("No transform received: %s", ex.what());
                        //ros::Duration(1.0).sleep();
                        return;
                    }
                }
            } else {
                observation_transform = latest_transform;
                observation_transform.header = msg->header;
            }

            observation_transform.header.frame_id = sensor_frame_name;
            observation_transform.child_frame_id = world_frame_name;
            words_pub.publish(toRosMsg(*wordObs3d, observation_transform));
        }
    }

public:
    explicit WordExtractorNode(ros::NodeHandle *nh)
            : tf_listener(tf_buffer)
            , wordAdapter(nh)
            , it(*nh) {
        std::string vocabulary_filename, texton_vocab_filename, image_topic_name, feature_descriptor_name, pc_topic_name, transform_topic_name;
        int num_surf, num_orb, color_cell_size, texton_cell_size;
        bool use_surf, use_hue, use_intensity, use_orb, use_texton;

        nh->param<string>("image", image_topic_name, "/camera/image_raw");
        nh->param<string>("transform", transform_topic_name, "");
        nh->param<string>("pc", pc_topic_name, "/point_cloud");
        nh->param<bool>("use_pc", use_pc, true);
        nh->param<bool>("publish_2d_words", publish_2d, false);
        nh->param<bool>("publish_3d_words", publish_3d, true);

        nh->param<double>("rate", rate, 0);

        nh->param<bool>("use_tf", use_tf, false);
        nh->param<string>("world_frame", world_frame_name, "map");
        nh->param<string>("sensor_frame", sensor_frame_name, "base_link");

        words_pub = nh->advertise<sunshine_msgs::WordObservation>("words", 1);
        words_2d_pub = nh->advertise<sunshine_msgs::WordObservation>("words_2d", 1);

        latest_transform = {};
        latest_transform.transform.rotation.w = 1; // Default no-op rotation
        transform_recvd = false;
        pc_recvd = false;

        auto const bagfile = nh->param<string>("image_bag", "");
        if (!bagfile.empty() && boost::filesystem::exists(bagfile)) {
            ROS_WARN("Extracting images from %s", bagfile.c_str());
            ros::Rate loop_rate(rate);
            rosbag::Bag bag;
            bag.open(bagfile);

            size_t i = 0;
            for (rosbag::MessageInstance const m: rosbag::View(bag, rosbag::TopicQuery(image_topic_name))) {
                sensor_msgs::Image::ConstPtr imgMsg = m.instantiate<sensor_msgs::Image>();
                if (imgMsg != nullptr) {
                    imageCallback(imgMsg);
                } else {
                    ROS_ERROR("Non-image message found in topic %s", image_topic_name.c_str());
                }
                i++;
                if (rate > 0) loop_rate.sleep();
            }
            bag.close();
            ROS_INFO("Extracted %lu images from rosbag.", i);
        } else {
            if (!bagfile.empty()) ROS_ERROR("Rosbag image bag set but file does not exist.");
            imageSub = it.subscribe(image_topic_name, 3, &WordExtractorNode::imageCallback, this);
            if (use_tf && !transform_topic_name.empty()) {
                transformSub = nh->subscribe<geometry_msgs::TransformStamped>(transform_topic_name,
                                                                              1,
                                                                              &WordExtractorNode::transformCallback,
                                                                              this);
            }
            if (use_pc) {
                depthSub = nh->subscribe<sensor_msgs::PointCloud2>(pc_topic_name, 1, &WordExtractorNode::pcCallback, this);
            }
        }
    }

    void spin() {
        if (rate <= 0) {
            ros::spin();
        } else {
            ros::Rate loop_rate(rate);
            while (ros::ok()) {
                ros::spinOnce();
                loop_rate.sleep();
            }
        }
    }
};
}

#endif //SUNSHINE_PROJECT_VISUAL_WORD_EXTRACTOR_HPP
