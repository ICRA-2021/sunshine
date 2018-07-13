#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "cv_bridge/cv_bridge.h"

//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>

#include "visualwords/image_source.hpp"
#include "visualwords/texton_words.hpp"
#include "visualwords/feature_words.hpp"
#include "visualwords/color_words.hpp"

#include "sunshine_msgs/WordObservation.h"

#include <iostream>
#include <boost/filesystem.hpp>

using namespace std;

namespace sunshine{

  static ros::Publisher words_pub;
  static MultiBOW multi_bow;
  
  void imageCallback(const sensor_msgs::ImageConstPtr& msg){

    cv_bridge::CvImageConstPtr img_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    
    cv::Mat img = img_ptr->image;

    //cv::imshow("Test",img);
    //cv::waitKey(3);
    
    WordObservation z = multi_bow(img);
    sunshine_msgs::WordObservation::Ptr sz(new sunshine_msgs::WordObservation);

    vector<int> pose(3);
    pose[0] = msg->header.seq;
    pose[1] = 0;
    pose[2] = 0;
    
    sz->source = z.source;
    sz->seq = msg->header.seq;
    sz->observation_pose = pose;
    sz->vocabulary_begin = z.vocabulary_begin;
    sz->vocabulary_size = z.vocabulary_size;
    sz->words = z.words;
    sz->word_pose = z.word_pose;
    sz->word_scale = z.word_scale;

    words_pub.publish(sz);
  }
}

int main(int argc, char** argv){
  // Setup ROS node
  ros::init(argc, argv, "word_extractor");
  ros::NodeHandle nhp("~");
  ros::NodeHandle nh("");

  char* data_root_c;
  data_root_c = getenv("ROSTPATH");
  std::string data_root;
  if (data_root_c!=NULL){
    cerr << "ROSTPATH: " << data_root_c << endl; //TODO: ROS_WARNING
    data_root = data_root_c;
  }

  std::string vocabulary_filename, texton_vocab_filename, image_topic_name, feature_descriptor_name;
  int num_surf, num_orb, color_cell_size, texton_cell_size;
  bool use_surf, use_hue, use_intensity, use_orb, use_texton;
  double img_scale;

  // Parse parameters
  double rate; //looping rate

  nhp.param<bool>("use_texton", use_texton, true);
  nhp.param<int>("num_texton", texton_cell_size, 64);
  nhp.param<string>("texton_vocab", texton_vocab_filename, data_root + "/share/visualwords/texton.vocabulary.baraka.1000.csv");

  nhp.param<bool>("use_orb", use_orb, true);
  nhp.param<int>("num_orb", num_orb, 1000);
  nhp.param<string>("vocab", vocabulary_filename, data_root + "/share/visualwords/orb_vocab/default.yml");

  nhp.param<bool>("use_hue", use_hue, true);
  nhp.param<bool>("use_intensity", use_intensity, true);
  nhp.param<int>("color_cell_size", color_cell_size, 32);

  nhp.param<bool>("use_surf", use_surf, false);
  nhp.param<int>("num_surf", num_surf, 1000);
  
  nhp.param<double>("scale", img_scale, 1.0);
  nhp.param<string>("image", image_topic_name, "/camera/image_raw");
  nhp.param<double>("rate", rate, 0);

  nhp.param<string>("feature_descriptor",feature_descriptor_name, "ORB");

  vector<string> feature_detector_names;
  vector<int> feature_sizes;

  if(use_surf){
    feature_detector_names.push_back("SURF");
    feature_sizes.push_back(num_surf);
  }
  if(use_orb){
    feature_detector_names.push_back("ORB");
    feature_sizes.push_back(num_orb);
  }

  if(use_surf || use_orb){
    sunshine::multi_bow.add(new LabFeatureBOW(0,
					      vocabulary_filename, 
					      feature_detector_names,
					      feature_sizes,
					      feature_descriptor_name,
					      img_scale));
  }

  if(use_hue || use_intensity){
    sunshine::multi_bow.add(new ColorBOW(0, color_cell_size, img_scale, use_hue, use_intensity));
  }

  if(use_texton){
    sunshine::multi_bow.add(new TextonBOW(0, texton_cell_size, img_scale, texton_vocab_filename));
  }
  
  image_transport::ImageTransport it(nhp);
  image_transport::Subscriber sub = it.subscribe(image_topic_name, 1, sunshine::imageCallback);

  sunshine::words_pub = nhp.advertise<sunshine_msgs::WordObservation>("words", 1);

  if(rate==0)
    ros::spin();
  else{
    ros::Rate loop_rate(rate);
    while (ros::ok()){
      ros::spinOnce();
      loop_rate.sleep();
    }
  }

  return 0;
}
