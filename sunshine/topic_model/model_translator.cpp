//
// Created by stewart on 2/3/20.
//

#include "model_translator.hpp"
#include "adrost_utils.hpp"
#include "sunshine_msgs/GetTopicModel.h"
#include "sunshine_msgs/SetTopicModel.h"

using namespace sunshine;

int main(int argc, char **argv) {
    ros::init(argc, argv, "model_translator");
    ros::NodeHandle nh("~");

    model_translator model(&nh);

    ros::MultiThreadedSpinner spinner;
    ROS_INFO("Spinning...");
    spinner.spin();
}

static match_results match_topics(std::string method, std::vector<Phi> const& topic_models) {
    if (method == "hungarian") {
        return sequential_hungarian_matching(topic_models);
    } else {
        ROS_ERROR("Unrecognized matching method: %s", method.c_str());
        return {};
    }
}

model_translator::model_translator(ros::NodeHandle *nh)
      : nh(nh) {
    std::string const target_nodes = nh->param<std::string>("target_nodes", "");
    std::stringstream ss(target_nodes);
    std::string node;
    while (std::getline(ss, node, ',')) {
        this->target_models.push_back(node);
        this->fetch_topic_clients.push_back(nh->serviceClient<GetTopicModel>("/" + node + "/get_topic_model", false));
        this->set_topic_clients.push_back(nh->serviceClient<SetTopicModel>("/" + node + "/set_topic_model", false));
    }
    this->match_method = nh->param<std::string>("match_method", "hungarian");
    this->match_period = nh->param<double>("match_period", -1);
    this->merge_period = nh->param<double>("merge_period", -1);

    if (this->match_period > 0) {
        this->match_publisher = nh->advertise<TopicMatches>("matches", 1);
        this->match_timer = nh->createTimer(ros::Duration(this->match_period), [this](ros::TimerEvent const &) {
            ROS_INFO("Beginning topic matching.");
            auto const topic_models = fetch_topic_models();
            auto correspondences = match_topics(this->match_method, {topic_models.begin(), topic_models.end()}).lifting;
            TopicMatches topic_matches;
            for (auto const& topic_model : topic_models) {
                topic_matches.robots.push_back(topic_model.id);
            }
            for (auto const& k_matches : correspondences) {
                topic_matches.Ks.push_back(k_matches.size());
                topic_matches.matches.insert(topic_matches.matches.end(), k_matches.begin(), k_matches.end());
                topic_matches.K_global = std::max(topic_matches.K_global, *std::max_element(k_matches.begin(), k_matches.end()) + 1);
            }
            match_publisher.publish(topic_matches);
            ROS_INFO("Finished topic matching.");
        });
    }

    if (this->merge_period > 0) {
        this->merge_timer = nh->createTimer(ros::Duration(this->merge_period), [this](ros::TimerEvent const &) {
            ROS_INFO("Beginning topic merging.");
            auto const topic_models = fetch_topic_models();
            auto correspondences = match_topics(this->match_method, {topic_models.begin(), topic_models.end()});
            TopicModel const global_model = TopicModel(merge_models(topic_models, correspondences));

            auto i = 0ul;
            for (auto& serviceClient : this->set_topic_clients) {
                SetTopicModel setTopicModel;
                setTopicModel.request.topic_model = global_model;
                if (!serviceClient.call(setTopicModel)) ROS_WARN("Failed to set topic model for robot %s!", target_models[i].c_str());
                i += 1;
            }
            ROS_INFO("Finished topic merging.");
        });
    }

    this->match_models_service = [this](MatchModelsRequest &req, MatchModelsResponse &resp){
        auto correspondences = match_topics(this->match_method, {req.topic_models.begin(), req.topic_models.end()}).lifting;
        resp = MatchModelsResponse();
        for (auto const& topic_model : req.topic_models) {
            resp.topic_matches.robots.push_back(topic_model.identifier);
        }
        for (auto const& k_matches : correspondences) {
            resp.topic_matches.Ks.push_back(k_matches.size());
            resp.topic_matches.matches.insert(resp.topic_matches.matches.end(), k_matches.begin(), k_matches.end());
            resp.topic_matches.K_global = std::max(resp.topic_matches.K_global, *std::max_element(k_matches.begin(), k_matches.end()));
        }
        return true;
    };
    this->topic_match_server = nh->advertiseService("match_topics", this->match_models_service);
}

std::vector<Phi> model_translator::fetch_topic_models() {
    ROS_INFO("Attempting to fetch %lu topic models", this->target_models.size());
    std::vector<Phi> topic_models;
    topic_models.reserve(this->target_models.size());
    for (auto& serviceClient : this->fetch_topic_clients) {
        GetTopicModel getTopicModel;
        if (serviceClient.call(getTopicModel)) topic_models.emplace_back(getTopicModel.response.topic_model);
    }
    ROS_INFO("Fetched %lu topic models", topic_models.size());
    return topic_models;
}
