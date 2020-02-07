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
#include "adrost_utils.hpp"
#include "csv.hpp"

namespace sunshine {
    using namespace sunshine_msgs;

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

      boost::function<bool(MatchModelsRequest &, MatchModelsResponse &)> match_models_service;
      std::vector<Phi> fetch_topic_models(bool pause_models = false);
      void broadcast_global_model(bool unpause_models);
      void pause_topic_models(bool new_pause_state);

      void update_global_model(std::vector<Phi> const &topic_models, match_results const &matches);

    public:
      explicit model_translator(ros::NodeHandle *nh);
    };
}

#endif //SUNSHINE_PROJECT_MODEL_TRANSLATOR_HPP
