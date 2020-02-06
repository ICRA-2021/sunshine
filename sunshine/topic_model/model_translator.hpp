//
// Created by stewart on 2/3/20.
//

#ifndef SUNSHINE_PROJECT_MODEL_TRANSLATOR_HPP
#define SUNSHINE_PROJECT_MODEL_TRANSLATOR_HPP

#include <ros/ros.h>
#include <sunshine_msgs/MergeModels.h>
#include <sunshine_msgs/MatchModels.h>
#include <list>

namespace sunshine {
    using namespace sunshine_msgs;

    struct Phi {
      std::string id;
      int K = 0, V = 0;
      std::vector<std::vector<int>> counts = {};
      std::vector<int> topic_weights = {};

      explicit Phi(std::string id)
            : id(id) {}

      Phi(std::string id, int K, int V, std::vector<std::vector<int>> counts, std::vector<int> topic_weights)
            : id(id)
            , K(K)
            , V(V)
            , counts(counts)
            , topic_weights(topic_weights) {
      }

      explicit Phi(TopicModel const &topic_model)
            : id(topic_model.identifier)
            , K(topic_model.K)
            , V(topic_model.V)
            , topic_weights(topic_model.topic_weights) {
          assert(*std::min_element(topic_weights.cbegin(), topic_weights.cend()) > 0);
          counts.reserve(K);
          for (auto i = 0ul; i < K; ++i) {
              counts.emplace_back(topic_model.phi.begin() + i * V,
                                  (i < K)
                                  ? topic_model.phi.begin() + (i + 1) * V
                                  : topic_model.phi.end());
              assert(counts[i].size() == V);
          }
      }

      explicit operator TopicModel() const {
          TopicModel topicModel;
          topicModel.K = this->K;
          topicModel.V = this->V;
          topicModel.identifier = this->id;
          topicModel.topic_weights = this->topic_weights;
          topicModel.phi.reserve(this->K * this->V);
          for (auto i = 0ul; i < K; ++i) {
              topicModel.phi.insert(topicModel.phi.end(), counts[i].begin(), counts[i].end());
          }
          assert(topicModel.phi.size() == K * V);
          return topicModel;
      }
    };

    class model_translator {
      ros::NodeHandle *nh;
      std::string match_method;
      double match_period;
      double merge_period;
      std::string save_model_path;

      std::vector<std::string> target_models;
      std::list<ros::ServiceClient> fetch_topic_clients;
      std::list<ros::ServiceClient> set_topic_clients;
      std::list<ros::ServiceClient> pause_topic_clients;

      ros::ServiceServer topic_match_server, topic_merge_server;
      ros::Timer match_timer, merge_timer;
      ros::Publisher match_publisher;

      Phi global_model;

      boost::function<bool(MatchModelsRequest &, MatchModelsResponse &)> match_models_service;
      std::vector<Phi> fetch_topic_models(bool pause_models = false);
      void broadcast_topic_model(TopicModel new_model, bool unpause_models = true);
      void pause_topic_models(bool new_pause_state);

    public:
      explicit model_translator(ros::NodeHandle *nh);
    };
}

#endif //SUNSHINE_PROJECT_MODEL_TRANSLATOR_HPP
