#ifndef TOPIC_MODEL_HPP
#define TOPIC_MODEL_HPP

#include <array>
#include <memory>
#include <ros/ros.h>
#include <rost/rost.hpp>
#include <sunshine_msgs/WordObservation.h>

#define POSEDIM 4

namespace sunshine {

typedef double_t WordDimType;
typedef int32_t CellDimType;
typedef std::array<CellDimType, POSEDIM> cell_pose_t;
typedef std::array<WordDimType, POSEDIM> word_pose_t;
typedef neighbors<cell_pose_t> neighbors_t;
typedef ROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t>> ROST_t;

class topic_model {
    static std::vector<std::string> const VALID_MAP_PPX_TYPES;
    ros::NodeHandle* nh;
    ros::Publisher scene_pub, global_perplexity_pub, global_surprise_pub, local_surprise_pub, topic_weights_pub, map_pub;
    ros::Subscriber word_sub;

    std::mutex wordsReceivedLock;
    std::chrono::steady_clock::time_point lastWordsAdded;
    mutable int consecutive_rate_violations = 0;

    std::string words_topic_name;
    int K, V, last_time = -1; //number of topic types, number of word types
    double cell_size_time, cell_size_space, k_alpha, k_beta, k_gamma, k_tau, p_refine_rate_local, p_refine_rate_global;
    CellDimType G_time, G_space;
    int num_threads, min_obs_refine_time, obs_queue_size;
    bool polled_refine, update_topic_model, publish_topics, publish_local_surprise, publish_global_surprise, publish_ppx, publish_map;
    std::string map_ppx_type;
    size_t last_refine_count;
    std::unique_ptr<ROST_t> rost;

    std::vector<CellDimType> observation_times; //list of all time seq ids observed thus far.
    std::vector<cell_pose_t> current_cell_poses, last_poses;
    sunshine_msgs::WordObservation topicObs;

    std::atomic<bool> stopWork;
    std::vector<std::shared_ptr<std::thread>> workers;

    void wait_for_processing() const;
    void words_callback(const sunshine_msgs::WordObservation::ConstPtr& words);
    void broadcast_topics() const;

public:
    topic_model(ros::NodeHandle* nh);
    ~topic_model();
};

static inline cell_pose_t toCellPose(word_pose_t const& word, double cell_size_time, double cell_size_space)
{
    return {
        static_cast<CellDimType>(word[0] / cell_size_time),
        static_cast<CellDimType>(word[1] / cell_size_space),
        static_cast<CellDimType>(word[2] / cell_size_space),
        static_cast<CellDimType>(word[3] / cell_size_space)
    };
}

static inline word_pose_t toWordPose(cell_pose_t const& cell, double cell_size_time, double cell_size_space)
{
    return {
        static_cast<WordDimType>(cell[0] * cell_size_time),
        static_cast<WordDimType>(cell[1] * cell_size_space),
        static_cast<WordDimType>(cell[2] * cell_size_space),
        static_cast<WordDimType>(cell[3] * cell_size_space)
    };
}
}

#endif // TOPIC_MODEL_HPP
