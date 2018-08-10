#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "cv_bridge/cv_bridge.h"

#include "sensor_msgs/PointCloud2.h"
#include "pcl_ros/point_cloud.h"
#include "pcl/io/pcd_io.h"

#include "visualwords/image_source.hpp"
#include "visualwords/texton_words.hpp"
#include "visualwords/feature_words.hpp"
#include "visualwords/color_words.hpp"

#include "sunshine_msgs/WordObservation.h"
#include "geometry_msgs/TransformStamped.h"

#include <iostream>
#include <boost/filesystem.hpp>

using namespace std;

namespace sunshine{

  static ros::Publisher words_pub;
  static MultiBOW multi_bow;
  static geometry_msgs::TransformStamped latest_transform;
  static bool transform_recvd; // TODO: smarter way of handling stale/missing poses
  static bool depth_recvd;
  static bool use_pc;
  static cv::Mat *depth_map;
  static std::string frame_id = "";

  void transformCallback(const geometry_msgs::TransformStampedConstPtr& msg){
    // Callback to handle world to sensor transform
    if (!transform_recvd) transform_recvd = true;
      
    latest_transform = *msg;
    frame_id = msg->child_frame_id;
  }

  void pcCallback(const sensor_msgs::PointCloud2ConstPtr& msg){
    // TODO: probably a better way to do this, assumes pixel coordinates and non-overlapping z coordinates
    // convert point cloud into a depth map
    // Source: https://answers.ros.org/question/136916/conversion-from-sensor_msgspointcloud2-to-pclpointcloudt/

      if (!transform_recvd){
        ROS_ERROR("No transformation received, ignoring point cloud");
        return;
      }

      if (!frame_id.empty() && frame_id != msg->header.frame_id) {
          ROS_ERROR("Point cloud frame id '%s' does not match transform frame '%s'. Ignoring point cloud.", msg->header.frame_id.c_str(), frame_id.c_str());
          return;
      }

    if (!depth_recvd){
      depth_map = new cv::Mat(msg->height, msg->width, CV_64FC1);
      depth_recvd = true;
    }
    
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_pcl(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pcl_pc2,*tmp_pcl);

    for (auto pt : tmp_pcl->points){
      depth_map->at<double>(cv::Point((int)pt.x,(int)pt.y)) = (double)pt.z;
    }
  }
  
  void imageCallback(const sensor_msgs::ImageConstPtr& msg){

    if (!transform_recvd){
      ROS_ERROR("No transformation received, observations will not be published");
      return;
    }

    if (!depth_recvd && use_pc) {
        ROS_ERROR("No depth map received, observations will not be published");
        return;
    }
    
    cv_bridge::CvImageConstPtr img_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    
    cv::Mat img = img_ptr->image;

    WordObservation z = multi_bow(img);
    sunshine_msgs::WordObservation::Ptr sz(new sunshine_msgs::WordObservation());

    size_t const num_words = z.words.size();
    size_t const poseDim = (use_pc) ? 3 : 2;
    
    sz->source = z.source;
    sz->seq = msg->header.seq;
    sz->observation_transform = latest_transform;
    sz->vocabulary_begin = z.vocabulary_begin;
    sz->vocabulary_size = z.vocabulary_size;
    sz->words = z.words;
    sz->word_pose.resize(num_words * poseDim);
    //std::copy(std::begin(z.word_pose), std::end(z.word_pose), std::begin(sz->word_pose));
    for(size_t i=0; i<num_words; ++i){
      int u,v;
      u = z.word_pose[i*2];
      v = z.word_pose[i*2+1];
      sz->word_pose[i*poseDim] = static_cast<double>(u);
      sz->word_pose[i*poseDim+1] = static_cast<double>(v);
      if (use_pc) {
        sz->word_pose[i*poseDim+2] = depth_map->at<double>(cv::Point(u, v));
      }
    }
    
    sz->word_scale.resize(num_words);
    sz->word_scale = z.word_scale;
    sz->header.frame_id = frame_id;

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

  std::string vocabulary_filename, texton_vocab_filename, image_topic_name,
    feature_descriptor_name, pc_topic_name, transform_topic_name;
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
  nhp.param<string>("transform", transform_topic_name, "/camera_world_transform");
  nhp.param<string>("pc", pc_topic_name, "/point_cloud");
  nhp.param<bool>("use_pc", sunshine::use_pc, false);
  
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

  if(use_texton){
    sunshine::multi_bow.add(new TextonBOW(0, texton_cell_size, img_scale, texton_vocab_filename));
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
  
  image_transport::ImageTransport it(nhp);
  image_transport::Subscriber sub = it.subscribe(image_topic_name, 1, sunshine::imageCallback);

  ros::Subscriber transformCallback = nhp.subscribe<geometry_msgs::TransformStamped>(transform_topic_name, 1, sunshine::transformCallback);
  ros::Subscriber depthCallback = nhp.subscribe<sensor_msgs::PointCloud2>(pc_topic_name, 1, sunshine::pcCallback);

  sunshine::words_pub = nhp.advertise<sunshine_msgs::WordObservation>("words", 1);

  sunshine::transform_recvd = false;
  sunshine::depth_recvd = false;

  if(rate <= 0)
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
