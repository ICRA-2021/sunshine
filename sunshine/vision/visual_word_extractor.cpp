#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "cv_bridge/cv_bridge.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "visualwords/image_source.hpp"
#include "visualwords/texton_words.hpp"
#include "visualwords/feature_words.hpp"
#include "visualwords/color_words.hpp"

#include "sunshine_msgs/WordObservation.h"

#include <iostream>
#include <boost/filesystem.hpp>

namespace sunshine{

  ros::Publisher words_pub;

  MultiBOW multi_bow;
  
  void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    cv_bridge::CvImageConstPtr img_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    
    cv::Mat img = img_ptr->image;

    WordObservation z = multi_bow(img);
    sunshine_msgs::WordObservation::Ptr sz(new sunshine_msgs::WordObservation);

    sz->source = z.source;
    sz->seq = z.seq;
    sz->vocabulary_begin = z.vocabulary_begin;
    sz->vocabulary_size = z.vocabulary_size;
    sz->observation_pose = z.observation_pose;
    sz->words = z.words;
    sz->word_pose = z.word_pose;
    sz->word_scale = z.word_scale;

    words_pub.publish(sz);
  }
}

int main(int argc, char** argv){
  
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

  std::string vocabname = data_root+"/share/visualwords/texton.vocabulary.baraka.1000.csv";

  TextonBOW texton_bow(0, 64, 1.0, vocabname);
  
  ColorBOW color_bow(0, 32, 1.0, true, true);
  
  std::vector<std::string> orb_feature_detector_names(1, std::string("ORB"));
  std::vector<int> orb_num_features(1, 1000) ;
  LabFeatureBOW orb_bow(0,
			data_root+"/share/visualwords/orb_vocab/default.yml", 
			orb_feature_detector_names,
			orb_num_features,
			"ORB",
			1.0);

  sunshine::multi_bow.add(&color_bow);
  sunshine::multi_bow.add(&texton_bow);
  sunshine::multi_bow.add(&orb_bow);
  
  image_transport::ImageTransport it(nhp);
  image_transport::Subscriber sub = it.subscribe("/camera/image_raw", 1, sunshine::imageCallback);

  sunshine::words_pub = nh.advertise<sunshine_msgs::WordObservation>("words", 1);

  ros::spin();
  
  return 0;
}
