//
// Created by stewart on 2/3/20.
//

#include "model_translator.hpp"
#include "adrost_utils.hpp"
#include "sunshine_msgs/GetTopicModel.h"
#include "sunshine_msgs/SetTopicModel.h"
#include "sunshine_msgs/Pause.h"
#include <fstream>
#include <utility>
#include <thread>

using namespace sunshine;

int main(int argc, char **argv) {
//    std::this_thread::sleep_for(std::chrono::seconds(5));
    ros::init(argc, argv, "model_translator");
    ros::NodeHandle nh("~");

    model_translator model(&nh);

    ros::MultiThreadedSpinner spinner;
    ROS_INFO("Spinning...");
    spinner.spin();
}

static match_results match_topics(std::string const &method, std::vector<Phi> const &topic_models) {
    if (method == "id") {
        return id_matching(topic_models);
    } else if (method == "hungarian") {
        return sequential_hungarian_matching(topic_models);
    } else {
        ROS_ERROR("Unrecognized matching method: %s", method.c_str());
        return {};
    }
}

static void merge_models(std::vector<sunshine::Phi> const &topic_models, match_results const &matches, Phi &global_model) {
    assert(!topic_models.empty());
    if (global_model.counts.empty() || global_model.K != matches.num_unique) {
        assert(global_model.counts.empty() || global_model.V == topic_models[0].V);
        global_model.K = matches.num_unique;
        global_model.V = topic_models[0].V;
        global_model.topic_weights.resize(global_model.K, 0);
        global_model.counts.resize(global_model.K, std::vector<int>(global_model.V, 0));
    }

    std::vector<std::vector<int>> const old_counts = global_model.counts;
    for (auto i = 0ul; i < topic_models.size(); ++i) {
        for (auto k2 = 0ul; k2 < topic_models[i].K; ++k2) {
            auto const &k1 = matches.lifting[i][k2];
            assert(k1 < matches.num_unique);
            assert(topic_models[i].V == global_model.V);

            for (auto v = 0ul; v < global_model.V; ++v) {
                int delta = topic_models[i].counts[k2][v] - old_counts[k2][v];
                ROS_WARN_COND(global_model.counts[k1][v] + delta < 0, "Delta implies a negative topic-word count!");
                delta = std::max(delta, -global_model.counts[k1][v]);
                global_model.counts[k1][v] += delta;
                ROS_ERROR_COND(global_model.counts[k1][v] < topic_models[i].counts[k2][v],
                               "The global count is less than the individual count!");
                global_model.topic_weights[k1] += delta;
            }
        }
    }
}

model_translator::model_translator(ros::NodeHandle *nh)
      : nh(nh)
      , global_model("global") {
    std::string const target_nodes = nh->param<std::string>("target_nodes", "");
    std::stringstream ss(target_nodes);
    std::string node;
    while (std::getline(ss, node, ',')) {
        this->target_models.push_back(node);
        this->fetch_topic_clients.push_back(nh->serviceClient<GetTopicModel>("/" + node + "/get_topic_model", false));
        this->set_topic_clients.push_back(nh->serviceClient<SetTopicModel>("/" + node + "/set_topic_model", false));
        this->pause_topic_clients.push_back(nh->serviceClient<Pause>("/" + node + "/pause_topic_model", false));
    }
    this->match_method = nh->param<std::string>("match_method", "hungarian");
    ROS_INFO("Using match method '%s'", this->match_method.c_str());
    this->match_period = nh->param<double>("match_period", -1);
    this->merge_period = nh->param<double>("merge_period", -1);
    this->save_model_path = nh->param<std::string>("save_model_path", "");

    if (this->match_period > 0) {
        this->match_publisher = nh->advertise<TopicMatches>("matches", 1);
        this->match_timer = nh->createTimer(ros::Duration(this->match_period), [this](ros::TimerEvent const &) {
            ROS_INFO("Beginning topic matching.");
            auto const topic_models = fetch_topic_models();
            auto correspondences = match_topics(this->match_method, {topic_models.begin(), topic_models.end()}).lifting;
            TopicMatches topic_matches;
            for (auto const &topic_model : topic_models) {
                topic_matches.robots.push_back(topic_model.id);
            }
            for (auto const &k_matches : correspondences) {
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
            auto const topic_models = fetch_topic_models(true);
            if (topic_models.size() < this->set_topic_clients.size()) {
                ROS_ERROR("Failed to fetch all models - skipping topic merging!");
                if (topic_models.size() > 0) pause_topic_models(false);
                return; // only merge if we have everything
            }
            auto const correspondences = match_topics(this->match_method, {topic_models.begin(), topic_models.end()});
            merge_models(topic_models, correspondences, global_model);
            broadcast_topic_model((TopicModel) global_model, true);
            ROS_INFO("Finished topic merging.");
        });
    }

    this->match_models_service = [this](MatchModelsRequest &req, MatchModelsResponse &resp) {
        auto correspondences = match_topics(this->match_method, {req.topic_models.begin(), req.topic_models.end()}).lifting;
        resp = MatchModelsResponse();
        for (auto const &topic_model : req.topic_models) {
            resp.topic_matches.robots.push_back(topic_model.identifier);
        }
        for (auto const &k_matches : correspondences) {
            resp.topic_matches.Ks.push_back(k_matches.size());
            resp.topic_matches.matches.insert(resp.topic_matches.matches.end(), k_matches.begin(), k_matches.end());
            resp.topic_matches.K_global = std::max(resp.topic_matches.K_global, *std::max_element(k_matches.begin(), k_matches.end()));
        }
        return true;
    };
    this->topic_match_server = nh->advertiseService("match_topics", this->match_models_service);
}

std::vector<Phi> model_translator::fetch_topic_models(bool pause_models) {
    Pause pause = {};
    pause.request.pause = pause_models;

    std::vector<Phi> topic_models;
    topic_models.reserve(this->target_models.size());
    auto fetchClient = this->fetch_topic_clients.begin();
    auto pauseClient = this->pause_topic_clients.begin();
    for (auto const &target_model : this->target_models) {
        if (!pause_models || pauseClient->call(pause)) {
            GetTopicModel getTopicModel = {};
            if (fetchClient->call(getTopicModel)) {
                topic_models.emplace_back(getTopicModel.response.topic_model);
                if (!save_model_path.empty()) {
                    std::string filename = save_model_path + "/" + std::to_string(ros::Time::now().sec) + "_" + target_model + ".bin";
                    std::fstream writer(filename, std::ios::out | std::ios::binary);
                    if (writer.good()) {
                        writer.write(reinterpret_cast<char *>(getTopicModel.response.topic_model.phi.data()),
                                     sizeof(decltype(getTopicModel.response.topic_model.phi)::value_type) / sizeof(char)
                                           * getTopicModel.response.topic_model.phi.size());
                        writer.close();
                    } else
                        ROS_WARN("Failed to save topic model to file %s", filename.c_str());
                }
            } else {
                ROS_WARN("Failed to fetch topic model from %s after pausing! Attempting to unpause.", target_model.c_str());
                pause.request.pause = false;
                if (!pauseClient->call(pause)) ROS_ERROR("Failed to unpause %s!", target_model.c_str());
                pause.request.pause = pause_models;
            }
        }
        fetchClient++;
        pauseClient++;
    }
    ROS_INFO("Fetched %lu of %lu topic models", topic_models.size(), this->target_models.size());
    return topic_models;
}

void model_translator::broadcast_topic_model(TopicModel new_model, bool unpause_models) {
    Pause pause = {};
    pause.request.pause = !unpause_models;

    SetTopicModel setTopicModel = {};
    setTopicModel.request.topic_model = std::move(new_model);

    std::vector<Phi> topic_models;
    topic_models.reserve(this->target_models.size());
    auto setClient = this->set_topic_clients.begin();
    auto pauseClient = this->pause_topic_clients.begin();
    for (auto const &target_model : this->target_models) {
        if (setClient->call(setTopicModel)) {
            if (unpause_models && !pauseClient->call(pause)) {
                ROS_ERROR("Failed to unpause topic model for %s!", target_model.c_str());
            }
        } else {
            ROS_ERROR("Failed to set topic model for %s!", target_model.c_str());
            pause.request.pause = false;
        }
        setClient++;
        pauseClient++;
    }
}

void model_translator::pause_topic_models(bool pause_models) {
    Pause pause = {};
    pause.request.pause = pause_models;
    auto pauseClient = this->pause_topic_clients.begin();
    for (auto const &target_model : this->target_models) {
        if (!pauseClient->call(pause)) {
            ROS_ERROR("Failed to %s topic model for %s!", (pause_models) ? "pause" : "unpause", target_model.c_str());
        }
    }
}
