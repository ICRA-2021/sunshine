#ifndef SUNSHINE_PROJECT_TOPIC_MODEL_NODE_HPP
#define SUNSHINE_PROJECT_TOPIC_MODEL_NODE_HPP

#include <array>
#include <memory>
#include <ros/ros.h>
#include <sunshine_msgs/WordObservation.h>
#include <sunshine_msgs/TopicMap.h>
#include <sunshine_msgs/SaveObservationModel.h>
#include <sunshine_msgs/GetTopicSummary.h>
#include <sunshine_msgs/GetTopicMap.h>
#include <sunshine_msgs/GetTopicModel.h>
#include <sunshine_msgs/SetTopicModel.h>
#include <sunshine_msgs/Pause.h>
#include "sunshine/rost_adapter.hpp"

#define POSEDIM 3

namespace sunshine {

class topic_model_node {
    static std::vector<std::string> const VALID_MAP_PPX_TYPES;
    ros::NodeHandle *nh;

    std::shared_ptr<std::thread> broadcast_thread;
    ROSTAdapter<POSEDIM> rostAdapter;
    std::unique_ptr<activity_manager::WriteToken> externalRostLock;

    std::string words_topic_name;
    int obs_queue_size;
    bool publish_topics, publish_local_surprise, publish_global_surprise, publish_ppx;
    float map_publish_period, save_topics_period;

    ros::Timer map_publish_timer, save_topics_timer;
    std::string save_topics_path;
    std::string map_ppx_type, current_source = "";
    std::chrono::time_point<std::chrono::steady_clock> start_refine_time, last_obs_time;

    ros::Publisher scene_pub, global_perplexity_pub, global_surprise_pub, local_surprise_pub, topic_weights_pub, map_pub;
    ros::Subscriber word_sub;

    ros::ServiceServer time_topic_server, cell_topic_server, topic_summary_server, topic_map_server;
    ros::ServiceServer get_topic_model_server, set_topic_model_server, pause_server;

    boost::function<bool(sunshine_msgs::SaveObservationModelRequest &req,
                         sunshine_msgs::SaveObservationModelResponse &)> const save_topics_by_time_csv;
    boost::function<bool(sunshine_msgs::SaveObservationModelRequest &,
                         sunshine_msgs::SaveObservationModelResponse &)> const save_topics_by_cell_csv;
    boost::function<bool(sunshine_msgs::GetTopicSummaryRequest &, sunshine_msgs::GetTopicSummaryResponse &)> const generate_topic_summary;
    boost::function<bool(sunshine_msgs::GetTopicMapRequest &, sunshine_msgs::GetTopicMapResponse &)> const get_topic_map;
    boost::function<bool(sunshine_msgs::GetTopicModelRequest &, sunshine_msgs::GetTopicModelResponse &)> const get_topic_model;
    boost::function<bool(sunshine_msgs::SetTopicModelRequest &, sunshine_msgs::SetTopicModelResponse &)> const set_topic_model;
    boost::function<bool(sunshine_msgs::PauseRequest &, sunshine_msgs::PauseResponse &)> const pause_topic_model;

    void words_callback(const sunshine_msgs::WordObservation::ConstPtr &wordMsg);

    void broadcast_topics(int obs_time, std::vector<ROSTAdapter<POSEDIM>::cell_pose_t> const &broadcast_poses);

  public:
#ifndef NDEBUG
    std::vector<std::vector<int>> externalTopicCounts = {}; // TODO Delete?
#endif

    explicit topic_model_node(ros::NodeHandle *nh);

    ~topic_model_node();

    ROSTAdapter<POSEDIM> const &get_adapter() const {
        return rostAdapter;
    }

    ROSTAdapter<POSEDIM> &get_adapter() {
        return rostAdapter;
    }

    sunshine_msgs::TopicMapPtr generate_topic_map(int obs_time) const;
};
}

#endif //SUNSHINE_PROJECT_TOPIC_MODEL_NODE_HPP
