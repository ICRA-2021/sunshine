#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "sunshine_msgs/WordObservation.h"
#include "sunshine_msgs/LocalSurprise.h"
#include "rost_visualize/draw_keypoints.hpp"
#include "rost_visualize/draw_local_surprise.hpp"

#include <iostream>
#include <algorithm>

std::map<unsigned, cv::Mat> image_cache;

using namespace std;

bool show_topics, show_local_surprise, show_perplexity, show_words, show_equalized;
string image_topic_name, words_topic_name, topic_topic_name, ppx_topic_name; //todo: rename topic model...

WordObservation msgToWordObservation(const sunshine_msgs::WordObservation::ConstPtr& z){
  WordObservation zz;
  zz.seq = z->seq;
  zz.source = z->source;
  zz.words = z->words;
  zz.word_pose = z->word_pose;
  zz.word_scale = z->word_scale;
  zz.vocabulary_begin = z->vocabulary_begin;
  zz.vocabulary_size = z->vocabulary_size;
  return zz;
}

void words_callback(const sunshine_msgs::WordObservation::ConstPtr& z){
  cv::Mat img = image_cache[z->seq];

  if(img.empty()) return;
  cv::Mat img_grey;
  cv::cvtColor(img,img_grey,CV_BGR2GRAY);
  cv::Mat img_grey_3c;
  cv::cvtColor(img_grey,img_grey_3c,CV_GRAY2BGR);

  WordObservation zz = msgToWordObservation(z);
  
  cv::Mat out_img = draw_keypoints(zz, img_grey_3c, 5);
  cv::imshow("Words", out_img);
  cv::waitKey(5); 
}

void topic_callback(const sunshine_msgs::WordObservation::ConstPtr& z){

  cv::Mat img = image_cache[z->seq];

  if(img.empty()) return;
  cv::Mat img_grey;
  cv::cvtColor(img,img_grey,CV_BGR2GRAY);
  cv::Mat img_grey_3c;
  cv::cvtColor(img_grey,img_grey_3c,CV_GRAY2BGR);
  
  WordObservation zz = msgToWordObservation(z);
  
  cv::Mat out_img = draw_keypoints(zz, img_grey_3c, 5);
  cv::imshow("Topics", out_img);
  cv::waitKey(5); 
}

void ppx_callback(const sunshine_msgs::LocalSurprise::ConstPtr& s_msg){
  
  std::map<std::array<int,3>, double> pose_surprise_map;

  // Gather poses and find the max and min values
  int max_x_idx = 0, max_y_idx = 0;
  
  assert(s_msg->surprise.size() == s_msg->surprise_poses.size()/2);
  for (int i = 0; i < s_msg->surprise_poses.size(); i += 2){
    std::array<int,3> pose;
    pose[0] = s_msg->seq;
    pose[1] = static_cast<int>(s_msg->surprise_poses[i]);
    pose[2] = static_cast<int>(s_msg->surprise_poses[i+1]);

    // todo: use std::max, not sure what was wrong before
    max_x_idx = (max_x_idx > pose[1]) ? max_x_idx : pose[1];
    max_y_idx = (max_y_idx > pose[2]) ? max_y_idx : pose[2];
    
    //max_x_idx = std::max(max_x_idx, s_msg->surprise_poses[i]);
    //max_y_idx = std::max(max_y_idx, s_msg->surprise_poses[i+1]);
    pose_surprise_map[pose] = s_msg->surprise[i/2];
  }

  // Initialize empty image of correct size
  cv::Mat ppx_img(max_x_idx + 1, max_y_idx + 1, CV_64F, cv::Scalar(0));
  
  // Draw on the image
  ppx_img = draw_pose_surprise<double>(pose_surprise_map, ppx_img, show_equalized);
  cv::Mat ppx_img_color(ppx_img.rows, ppx_img.cols, CV_8UC3, cv::Scalar(0,0,0));
  ppx_img_color = colorize(ppx_img, cv::Vec3b(0,0,255), cv::Vec3b(255,255,255));
  
  // Add it to the existing image
  cv::Mat img = image_cache[s_msg->seq];

  if (img.empty()) return;
  
  cv::Mat ppx_img_full;
  cv::resize(ppx_img_color, ppx_img_full, img.size());
  cv::addWeighted(img, 0.5, ppx_img_full, 0.9, 0.0, ppx_img_full);

  cv::imshow("Perplexity", ppx_img_full);
  cv::waitKey(5); 
}

void image_callback(const sensor_msgs::ImageConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

  image_cache[msg->header.seq] = cv_ptr->image.clone();

  if (image_cache.size() > 30){
    image_cache.erase(image_cache.begin());
  }
  
  //cv::imshow("Image", cv_ptr->image);
  //cv::waitKey(5);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "viewer");
  ros::NodeHandle nhp("~");
  ros::NodeHandle nh("");

  nhp.param<bool>("show_topics", show_topics, true);
  nhp.param<bool>("show_words", show_words, true);
  nhp.param<bool>("show_perplexity", show_perplexity, false);
  nhp.param<bool>("show_equalized", show_equalized, true);
  
  nhp.param<string>("words_topic", words_topic_name, "/words");
  nhp.param<string>("topics_topic", topic_topic_name, "/topics");
  nhp.param<string>("ppx_topic", ppx_topic_name, "/cell_perplexity");
  nhp.param<string>("image", image_topic_name, "/camera/image_raw");

  image_transport::ImageTransport it(nhp);
  image_transport::Subscriber img_sub = it.subscribe(image_topic_name, 1, image_callback);

  /*
  ros::Subscriber word_sub = nhp.subscribe(words_topic_name,1,words_callback);
  ros::Subscriber topic_sub = nhp.subscribe(topic_topic_name,1,topic_callback);
  ros::Subscriber ppx_sub = nhp.subscribe(ppx_topic_name,1,ppx_callback);
  */

  ros::Subscriber word_sub, topic_sub, ppx_sub;

  if (show_topics) topic_sub = nhp.subscribe(topic_topic_name,1,topic_callback);
  if (show_words) word_sub = nhp.subscribe(words_topic_name,1,words_callback);
  if (show_perplexity) ppx_sub = nhp.subscribe(ppx_topic_name,1,ppx_callback);
  
  ros::spin();

  return 0;
}
