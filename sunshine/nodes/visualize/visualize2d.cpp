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
#include "sunshine/common/utils.hpp"

#include <iostream>
#include <algorithm>


using namespace std;
using namespace sunshine;

static map<unsigned, cv::Mat> image_cache;
float scale;
static int cache_size;
static bool show_topics, show_perplexity, show_words, show_equalized;
static string image_topic_name, words_topic_name, topic_topic_name, ppx_topic_name; //todo: rename topic model...

WordObservation msgToWordObservation(const sunshine_msgs::WordObservation::ConstPtr& z){
  WordObservation zz;
  zz.seq = z->seq;
  zz.source = z->source;
  zz.words = z->words;

  zz.word_pose.resize(z->word_pose.size());
  std::copy(std::begin(z->word_pose), std::end(z->word_pose), std::begin(zz.word_pose));
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

  if (zz.words.size() > 0) {
    cv::Mat out_img = draw_keypoints(zz, img_grey_3c, 5);
    cv::resize(out_img, out_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::imshow("Words", out_img);
  } else {
    cv::imshow("Words", img_grey_3c);
  }
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
  cv::resize(out_img, out_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
  cv::imshow("Topics", out_img);
  cv::waitKey(5); 
}

void ppx_callback(const sunshine_msgs::LocalSurprise::ConstPtr& s_msg){
  // Create and normalize the image
  cv::Mat ppx_img = toMat<int32_t, double, float>(s_msg->surprise_poses, s_msg->surprise);
  cv::Scalar mean_ppx, stddev_ppx;
  cv::meanStdDev(ppx_img, mean_ppx, stddev_ppx);
  ppx_img = ppx_img - mean_ppx.val[0];
  ppx_img.convertTo(ppx_img, CV_32F, 1. / (2. * stddev_ppx.val[0]), 0.5);
  
  // Draw on the image
  cv::Mat ppx_img_color = colorize(ppx_img, cv::Vec3b(0,0,255), cv::Vec3b(255,255,255));
  
  // Add it to the existing image
  cv::Mat img = image_cache[s_msg->seq];

  if (img.empty()) return;
  
  cv::Mat ppx_img_full;
  cv::resize(ppx_img_color, ppx_img_full, img.size());
  cv::addWeighted(img, 0.5, ppx_img_full, 0.9, 0.0, ppx_img_full);

  cv::resize(ppx_img_full, ppx_img_full, cv::Size(), scale, scale, cv::INTER_LINEAR);
  cv::imshow("Perplexity", ppx_img_full);
  cv::waitKey(5); 
}

void image_callback(const sensor_msgs::ImageConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

  image_cache[msg->header.seq] = cv_ptr->image.clone();

  if (image_cache.size() > cache_size){
    image_cache.erase(image_cache.begin());
  }
  
  //cv::imshow("Image", cv_ptr->image);
  //cv::waitKey(5);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "viewer");
  ros::NodeHandle nhp("~");
  ros::NodeHandle nh("");

  nhp.param<float>("scale", scale, 1.0);
  nhp.param<int>("cache_size", cache_size, 100);
  nhp.param<bool>("show_topics", show_topics, true);
  nhp.param<bool>("show_words", show_words, true);
  nhp.param<bool>("show_perplexity", show_perplexity, false);
//  nhp.param<bool>("show_equalized", show_equalized, true);
  
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
