#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"

#include "sunshine_msgs/WordObservation.h"
#include "rost_visualize/draw_keypoints.hpp"

#include <iostream>
#include <algorithm>

std::map<unsigned, cv::Mat> image_cache;

using namespace std;

void image_callback(const sensor_msgs::ImageConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "viewer");
  ros::NodeHandle *nhp = new ros::NodeHandle("~");
  ros::NodeHandle *nh = new ros::NodeHandle("");

  bool show_topics, show_local_surprise, show_perplexity;
  string image_topic_name;
  nhp->param<bool>("topics", show_topics, true);
  nhp->param<bool>("local_surprise", show_local_surprise, true);
  nhp->param<string>("image", image_topic_name, "/image");
  nhp->param<bool>("perplexity", show_perplexity, true);
  
  return 0;
}
