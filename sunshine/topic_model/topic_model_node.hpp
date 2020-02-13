#ifndef TOPIC_MODEL_HPP
#define TOPIC_MODEL_HPP

#include <array>
#include <memory>
#include <ros/ros.h>
#include <rost/hlda.hpp>
#include <rost/rost.hpp>
#include <sunshine_msgs/WordObservation.h>
#include <sunshine_msgs/TopicMap.h>
#include <sunshine_msgs/SaveObservationModel.h>
#include <sunshine_msgs/GetTopicSummary.h>
#include <sunshine_msgs/GetTopicMap.h>
#include <sunshine_msgs/GetTopicModel.h>
#include <sunshine_msgs/SetTopicModel.h>
#include <sunshine_msgs/Pause.h>

#define POSEDIM 4

namespace sunshine {

typedef double_t WordDimType;
typedef int32_t CellDimType;
typedef std::array<CellDimType, POSEDIM> cell_pose_t;
typedef std::array<WordDimType, POSEDIM> word_pose_t;
typedef neighbors<cell_pose_t> neighbors_t;
typedef warp::SpatioTemporalTopicModel<cell_pose_t, neighbors_t, hash_container<cell_pose_t>> ROST_t;

class topic_model_node {
    static std::vector<std::string> const VALID_MAP_PPX_TYPES;
    ros::NodeHandle *nh;
    ros::Publisher scene_pub, global_perplexity_pub, global_surprise_pub, local_surprise_pub, topic_weights_pub, map_pub;
    ros::Subscriber word_sub;

    ros::ServiceServer time_topic_server, cell_topic_server, topic_summary_server, topic_map_server;
    ros::ServiceServer get_topic_model_server, set_topic_model_server, pause_server;

    mutable std::mutex wordsReceivedLock;
    std::chrono::steady_clock::time_point lastWordsAdded;
    mutable int consecutive_rate_violations = 0;

    std::string words_topic_name;
    int K, V, last_time = -1; //number of topic types, number of word types
    std::array<double, POSEDIM> cell_size;
    double k_alpha, k_beta, k_gamma, k_tau, p_refine_rate_local, p_refine_rate_global;
    CellDimType G_time, G_space;
    int num_threads, min_obs_refine_time, obs_queue_size;
    bool polled_refine, update_topic_model, publish_topics, publish_local_surprise, publish_global_surprise, publish_ppx;
    float map_publish_period, save_topics_period;
    ros::Timer map_publish_timer, save_topics_timer;
    std::string save_topics_path;
    std::string map_ppx_type;
    size_t last_refine_count;
    std::unique_ptr<ROST_t> rost;
    std::unique_ptr<activity_manager::WriteToken> externalRostLock;

    std::vector<CellDimType> observation_times; //list of all time seq ids observed thus far.
    std::vector<cell_pose_t> current_cell_poses, last_poses;
    std::string current_source;
    sunshine_msgs::WordObservation topicObs;

    std::atomic<bool> stopWork;
    std::vector<std::shared_ptr<std::thread>> workers;
    std::shared_ptr<std::thread> broadcast_thread;

    boost::function<bool(sunshine_msgs::SaveObservationModelRequest &req,
                         sunshine_msgs::SaveObservationModelResponse &)> const save_topics_by_time_csv;
    boost::function<bool(sunshine_msgs::SaveObservationModelRequest &,
                         sunshine_msgs::SaveObservationModelResponse &)> const save_topics_by_cell_csv;
    boost::function<bool(sunshine_msgs::GetTopicSummaryRequest &, sunshine_msgs::GetTopicSummaryResponse &)> const generate_topic_summary;
    boost::function<bool(sunshine_msgs::GetTopicMapRequest &, sunshine_msgs::GetTopicMapResponse &)> const get_topic_map;
    boost::function<bool(sunshine_msgs::GetTopicModelRequest &, sunshine_msgs::GetTopicModelResponse &)> const get_topic_model;
    boost::function<bool(sunshine_msgs::SetTopicModelRequest &, sunshine_msgs::SetTopicModelResponse &)> const set_topic_model;
    boost::function<bool(sunshine_msgs::PauseRequest &, sunshine_msgs::PauseResponse &)> const pause_topic_model;

    void wait_for_processing() const;

    void words_callback(const sunshine_msgs::WordObservation::ConstPtr &words);

    void broadcast_topics(const int obs_time, const std::vector<cell_pose_t> &broadcast_poses) const;

  public:
#ifndef NDEBUG
    std::vector<std::vector<int>> externalTopicCounts = {}; // TODO Delete?
#endif

    explicit topic_model_node(ros::NodeHandle *nh);

    ~topic_model_node();

    std::map<CellDimType, std::vector<int>> get_topics_by_time() const;

    std::map<cell_pose_t, std::vector<int>> get_topics_by_cell() const;

    decltype(K) get_num_topics() const {
        return K;
    }

    decltype(last_time) get_last_observation_time() const {
        return last_time;
    }

    ROST_t const &get_rost() const {
        return *rost;
    }

    ROST_t &get_rost() {
        return *rost;
    }

    sunshine_msgs::TopicMapPtr generate_topic_map(int obs_time) const;
};

static inline cell_pose_t toCellId(word_pose_t const &word, std::array<double, POSEDIM> cell_size) {
    static_assert(std::numeric_limits<CellDimType>::max() <= std::numeric_limits<WordDimType>::max(),
                  "Word dim type must be larger than cell dim type!");
    return {static_cast<CellDimType>(std::fmod(word[0] / cell_size[0], std::numeric_limits<CellDimType>::max())),
          static_cast<CellDimType>(std::fmod(word[1] / cell_size[1], std::numeric_limits<CellDimType>::max())),
          static_cast<CellDimType>(std::fmod(word[2] / cell_size[2], std::numeric_limits<CellDimType>::max())),
          static_cast<CellDimType>(std::fmod(word[3] / cell_size[3], std::numeric_limits<CellDimType>::max()))};
}

static inline word_pose_t toWordPose(cell_pose_t const &cell, std::array<double, POSEDIM> cell_size) {
    return {static_cast<WordDimType>(cell[0] * cell_size[0]), static_cast<WordDimType>(cell[1] * cell_size[1]),
          static_cast<WordDimType>(cell[2] * cell_size[2]), static_cast<WordDimType>(cell[3] * cell_size[3])};
}
}

#endif // TOPIC_MODEL_HPP
