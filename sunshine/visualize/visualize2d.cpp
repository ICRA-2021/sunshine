#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "sunshine_msgs/WordObservation.h"
#include "rost_visualize/draw_keypoints.hpp"

#include <iostream>
#include <algorithm>

std::map<unsigned, cv::Mat> image_cache;

using namespace std;

bool show_topics, show_local_surprise, show_perplexity;
string image_topic_name, words_topic_name, topic_topic_name; //todo: rename topic model...

void words_callback(const sunshine_msgs::WordObservation::ConstPtr& z){
  cv::Mat img = image_cache[z->seq];

  if(img.empty()) return;
  cv::Mat img_grey;
  cv::cvtColor(img,img_grey,CV_BGR2GRAY);
  cv::Mat img_grey_3c;
  cv::cvtColor(img_grey,img_grey_3c,CV_GRAY2BGR);

  WordObservation zz;
  zz.seq = z->seq;
  zz.source = z->source;
  zz.words = z->words;
  zz.word_pose = z->word_pose;
  zz.word_scale = z->word_scale;
  zz.vocabulary_begin = z->vocabulary_begin;
  zz.vocabulary_size = z->vocabulary_size;
  
  cv::Mat out_img = draw_keypoints(zz, img_grey_3c, 5);
  cv::imshow("z->source", out_img);
  cv::waitKey(5); 
}

void image_callback(const sensor_msgs::ImageConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

  image_cache[msg->header.seq] = cv_ptr->image.clone();
  if (image_cache.size() > 100){
    image_cache.erase(image_cache.begin());
  }
  
  cv::imshow("Image", cv_ptr->image);
  cv::waitKey(5);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "viewer");
  ros::NodeHandle nhp("~");
  ros::NodeHandle nh("");

  nhp.param<bool>("show_topics", show_topics, true);
  nhp.param<bool>("show_words", show_topics, true);
  nhp.param<string>("words_topic", words_topic_name, "/words");
  nhp.param<string>("topic_topic", topic_topic_name, "/topics");
  
  nhp.param<bool>("local_surprise", show_local_surprise, true);
  nhp.param<string>("image", image_topic_name, "/camera/image_raw");
  nhp.param<bool>("perplexity", show_perplexity, true);

  image_transport::ImageTransport it(nhp);
  image_transport::Subscriber img_sub = it.subscribe(image_topic_name, 1, image_callback);
  ros::Subscriber word_sub = nhp.subscribe(words_topic_name,1,words_callback);

  ros::spin();

  return 0;
}
