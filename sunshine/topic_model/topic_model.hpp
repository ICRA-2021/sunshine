#ifndef TOPIC_MODEL_HPP
#define TOPIC_MODEL_HPP

#include <array>
#include <memory>
#include <ros/ros.h>
#include <rost/rost.hpp>
#include <sunshine_msgs/LocalSurprise.h>
#include <sunshine_msgs/Perplexity.h>
#include <sunshine_msgs/TopicWeights.h>
#include <sunshine_msgs/WordObservation.h>

typedef std::array<int, 3> pose_t;
typedef ROST<pose_t, neighbors<pose_t>, hash_container<pose_t>> ROST_t;

class topic_model {
    ros::NodeHandle* nh;
    ros::Publisher topics_pub, perplexity_pub, cell_perplexity_pub, topic_weights_pub;
    ros::Subscriber word_sub;

    int K, V, cell_width; //number of topic types, number of word types
    double k_alpha, k_beta, k_gamma, k_tau, p_refine_last_observation;
    int G_time, G_space, num_threads, observation_size, last_time;
    bool polled_refine;
    size_t last_refine_count;
    std::unique_ptr<ROST_t> rost;

    map<size_t, set<pose_t>> cellposes_for_time; //list of all poses observed at a given time
    map<pose_t, vector<int>> worddata_for_pose; //stores [pose]-> {x_i,y_i,scale_i,.....} for the current time
    vector<int> observation_times; //list of all time seq ids observed thus far.

    std::atomic<bool> stopWork;
    std::vector<std::shared_ptr<std::thread>> workers;

    void words_callback(const sunshine_msgs::WordObservation::ConstPtr& words);
    void broadcast_topics();

public:
    topic_model(ros::NodeHandle* nh);
    ~topic_model();
};

#endif // TOPIC_MODEL_HPP
