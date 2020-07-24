//
// Created by stewart on 2/3/20.
//

#include "model_translator.hpp"
#include "sunshine_msgs/GetTopicModel.h"
#include "sunshine_msgs/SetTopicModel.h"
#include "sunshine_msgs/Pause.h"
#include "sunshine/common/ros_conversions.hpp"
#include <fstream>
#include <thread>
#include <numeric>

using namespace sunshine;
using namespace sunshine_msgs;

int main(int argc, char **argv) {
//    std::this_thread::sleep_for(std::chrono::seconds(5));
    ros::init(argc, argv, "model_translator");
    ros::NodeHandle nh("~");

    model_translator model(&nh);

    ros::MultiThreadedSpinner spinner;
    ROS_INFO("Spinning...");
    spinner.spin();
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
    this->match_method = nh->param<std::string>("match_method", "");
    ROS_INFO("Using match method '%s'", this->match_method.c_str());
    this->match_period = nh->param<double>("match_period", -1);
    this->merge_period = nh->param<double>("merge_period", -1);
    this->save_model_path = nh->param<std::string>("save_model_path", "");
    this->stats_path = nh->param<std::string>("stats_path", "");
    if (!stats_path.empty()) {
        this->stats_writer = std::make_unique<csv_writer<>>(stats_path);
        csv_row<> header{};
        header.append("Total # of Topics");
        header.append("SSD");
        header.append("Cluster Size");
        header.append("Matched Mean-Square Cluster Distance");
        header.append("Matched Silhouette Index");
        stats_writer->write_header(header);
    }

    if (this->match_period > 0) {
        ROS_INFO("Enabled matching with period %f", this->match_period);
        this->match_publisher = nh->advertise<TopicMatches>("matches", 1);
        this->match_timer = nh->createTimer(ros::Duration(this->match_period), [this](ros::TimerEvent const &) {
            ROS_INFO("Beginning topic matching.");
            auto const topic_models = fetch_topic_models();
            auto const correspondences = match_topics(this->match_method, {topic_models.begin(), topic_models.end()});
            TopicMatches topic_matches;
            for (auto const &topic_model : topic_models) {
                topic_matches.robots.push_back(topic_model.id);
            }
            for (auto const &k_matches : correspondences.lifting) {
                topic_matches.Ks.push_back(k_matches.size());
                topic_matches.matches.insert(topic_matches.matches.end(), k_matches.begin(), k_matches.end());
                topic_matches.K_global = std::max(topic_matches.K_global, *std::max_element(k_matches.begin(), k_matches.end()) + 1);
            }
            match_publisher.publish(topic_matches);
            if (this->stats_writer) {
                csv_row<> row{};
                row.append(correspondences.num_unique);
                row.append(correspondences.ssd);
                match_scores const scores(topic_models, correspondences.lifting, normed_dist_sq<double>);
                row.append(scores.cluster_sizes);
                row.append(scores.mscd);
                row.append(scores.silhouette);
                stats_writer->write_row(row);
                stats_writer->flush();
            }
            ROS_INFO("Finished topic matching.");
        });
    }

    if (this->merge_period > 0) {
        ROS_INFO("Enabled merging with period %f", this->merge_period);
        this->merge_timer = nh->createTimer(ros::Duration(this->merge_period), [this](ros::TimerEvent const &) {
            ROS_INFO("Beginning topic merging.");
            auto const topic_models = fetch_topic_models(true);
            if (topic_models.size() < this->set_topic_clients.size()) {
                ROS_ERROR("Failed to fetch all models - skipping topic merging!");
                if (!topic_models.empty()) pause_topic_models(false);
                return; // only merge if we have everything
            }
            auto const correspondences = match_topics(this->match_method, {topic_models.begin(), topic_models.end()});
            update_global_model(topic_models, correspondences);
            broadcast_global_model(true);
            if (this->stats_writer) {
                csv_row<> row{};
                row.append(correspondences.num_unique);
                row.append(correspondences.ssd);
                match_scores const scores(topic_models, correspondences.lifting, jensen_shannon_dist<double>);
                row.append(scores.cluster_sizes);
                row.append(scores.mscd);
                row.append(scores.silhouette);
                stats_writer->write_row(row);
                stats_writer->flush();
            }
            ROS_INFO("Finished topic merging.");
        });
    }

    this->match_models_service = [this](MatchModelsRequest &req, MatchModelsResponse &resp) {
        std::vector<Phi> topic_models;
        topic_models.reserve(req.topic_models.size());
        for (auto const &msg : req.topic_models) topic_models.push_back(fromRosMsg(msg));

        auto correspondences = match_topics(this->match_method, topic_models).lifting;
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
                topic_models.push_back(fromRosMsg(getTopicModel.response.topic_model));
                if (!save_model_path.empty()) {
                    std::string filename = save_model_path + "/" + std::to_string(ros::Time::now().sec) + "_"
                          + std::to_string(static_cast<int>(ros::Time::now().nsec / 1E6)) + "_" + target_model + ".bin";
                    std::fstream writer(filename, std::ios::out | std::ios::binary);
                    if (writer.good()) {
                        writer.write(reinterpret_cast<char *>(getTopicModel.response.topic_model.phi.data()),
                                     sizeof(decltype(getTopicModel.response.topic_model.phi)::value_type) / sizeof(char)
                                           * getTopicModel.response.topic_model.phi.size());
                        writer.close();
                    } else {
                        ROS_WARN("Failed to save topic model to file %s", filename.c_str());
                    }
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

void model_translator::update_global_model(std::vector<Phi> const &topic_models, match_results const &matches) {
    auto const total_weight = [](std::vector<int> const &vec) { return std::accumulate(vec.begin(), vec.end(), 0); };
    assert(!topic_models.empty());
    assert(total_weight(global_model.topic_weights) == total_num_observations);

    if (global_model.counts.empty() || global_model.K != matches.num_unique) {
        assert(global_model.counts.empty() || global_model.V == topic_models[0].V);
        global_model.K = matches.num_unique;
        global_model.V = topic_models[0].V;
        global_model.topic_weights.resize(matches.num_unique, 0);
        global_model.counts.resize(matches.num_unique, std::vector<int>(global_model.V, 0));
    }
    assert(total_weight(global_model.topic_weights) == total_num_observations);

//    auto const old_weights = global_model.topic_weights;
    std::vector<std::vector<int>> const old_counts = global_model.counts; // DO NOT use copy constructor!
    for (auto i = 0ul; i < topic_models.size(); ++i) {
        std::vector<int> weight_ref = topic_models[i].topic_weights;
        for (auto k2 = 0ul; k2 < topic_models[i].K; ++k2) {
            auto const &k1 = matches.lifting[i][k2];
            assert(k1 < matches.num_unique);
            assert(topic_models[i].V == global_model.V);

            for (auto v = 0ul; v < global_model.V; ++v) {
                int delta = topic_models[i].counts[k2][v] - old_counts[k2][v];
                assert(k2 == k1);
                if (delta < -global_model.counts[k1][v]) ROS_ERROR_THROTTLE(1, "Delta implies a negative topic-word count!");
                else if (delta < 0 && match_method != "id") {
                    ROS_WARN_THROTTLE(1, "Delta negative -- setting to 0 (see Doherty IROS'18)");
                    delta = 0;
                }
                global_model.counts[k1][v] += delta;
                global_model.topic_weights[k1] += delta;
//                weight_ref[k2] -= delta; assert(weight_ref[k2] >= 0);
            }
        }
//        ROS_INFO("Total weight ref after removing deltas: %d", total_weight(weight_ref)); assert(total_weight(weight_ref) == total_num_observations);
    }
    auto const total_out = total_weight(global_model.topic_weights);
    ROS_INFO("Added %li observations to global topic model.", total_out - total_num_observations);
#ifndef NDEBUG
    auto const total_in = std::accumulate(topic_models.begin(),
                                          topic_models.end(),
                                          0,
                                          [total_weight](int left, Phi const &right) { return left + total_weight(right.topic_weights); });
    ROS_INFO("Detected %li new observations.", total_in - total_num_observations * topic_models.size());
    ROS_INFO("%d in, %d out based on %li previous observations and %li new observations",
             total_in,
             total_out,
             total_num_observations,
             total_in - total_num_observations * topic_models.size());
    assert(total_in - total_num_observations * topic_models.size() == total_out - total_num_observations);
#endif
    total_num_observations = total_out;
}

void model_translator::broadcast_global_model(bool unpause_models) {
    Pause pause = {};
    pause.request.pause = !unpause_models;

    SetTopicModel setTopicModel = {};
    setTopicModel.request.topic_model = toRosMsg(global_model);

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
            ROS_ERROR("Failed to %s topic model for %s!",
                      (pause_models)
                      ? "pause"
                      : "unpause",
                      target_model.c_str());
        }
    }
}

model_translator::~model_translator() {
    merge_timer.stop();
    match_timer.stop();
    pause_topic_models(false);
}
