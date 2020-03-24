//
// Created by stewart on 2/3/20.
//

#ifndef SUNSHINE_PROJECT_MODEL_TRANSLATOR_HPP
#define SUNSHINE_PROJECT_MODEL_TRANSLATOR_HPP

#include <ros/ros.h>
#include <sunshine_msgs/MergeModels.h>
#include <sunshine_msgs/MatchModels.h>
#include <list>
#include <utility>
#include "sunshine/common/adrost_utils.hpp"
#include "sunshine/common/csv.hpp"

namespace sunshine {
    class model_translator {
      ros::NodeHandle *nh;
      std::string match_method;
      double match_period;
      double merge_period;
      std::string save_model_path, stats_path;
      std::unique_ptr<csv_writer<>> stats_writer;

      std::vector<std::string> target_models;
      std::list<ros::ServiceClient> fetch_topic_clients;
      std::list<ros::ServiceClient> set_topic_clients;
      std::list<ros::ServiceClient> pause_topic_clients;

      ros::ServiceServer topic_match_server, topic_merge_server;
      ros::Timer match_timer, merge_timer;
      ros::Publisher match_publisher;

      Phi global_model;
      long total_num_observations = 0;

      boost::function<bool(sunshine_msgs::MatchModelsRequest &, sunshine_msgs::MatchModelsResponse &)> match_models_service;
      std::vector<Phi> fetch_topic_models(bool pause_models = false);
      void broadcast_global_model(bool unpause_models);
      void pause_topic_models(bool new_pause_state);

      void update_global_model(std::vector<Phi> const &topic_models, match_results const &matches);

    public:
      explicit model_translator(ros::NodeHandle *nh);
      ~model_translator();
    };
}

#endif //SUNSHINE_PROJECT_MODEL_TRANSLATOR_HPP
