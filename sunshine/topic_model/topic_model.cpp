#include "topic_model.hpp"

#include <opencv2/core.hpp>
#include <rost/refinery.hpp>

using namespace sunshine;
using namespace sunshine_msgs;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "topic_model");
    ros::NodeHandle nh("~");

    topic_model model(&nh);

    ros::MultiThreadedSpinner spinner;
    ROS_INFO("Spinning...");
    spinner.spin();
}

topic_model::topic_model(ros::NodeHandle* nh)
    : nh(nh)
{
    nh->param<int>("K", K, 64); // number of topics
    nh->param<int>("V", V, 1500); // vocabulary size
    nh->param<double>("alpha", k_alpha, 0.1);
    nh->param<double>("beta", k_beta, 0.1);
    nh->param<double>("gamma", k_gamma, 0.0);
    nh->param<double>("tau", k_tau, 2.0); // beta(1,tau) is used to pick cells for global refinement
    nh->param<int>("observation_size", observation_size, 64); // number of cells in an observation
    nh->param<double>("p_refine_last_observation", p_refine_last_observation, 0.5); // probability of refining last observation
    nh->param<int>("num_threads", num_threads, 2); // beta(1,tau) is used to pick cells for refinement
    nh->param<int>("cell_width", cell_width, 64);
    nh->param<int>("G_time", G_time, 4);
    nh->param<int>("G_space", G_space, 1);
    nh->param<bool>("polled_refine", polled_refine, false);
    nh->param<bool>("update_topic_model", update_topic_model, true);

    ROS_INFO("Starting online topic modelling with parameters: K=%u, alpha=%f, beta=%f, gamma=%f tau=%f", K, k_alpha, k_beta, k_gamma, k_tau);

    scene_pub = nh->advertise<WordObservation>("topics", 10);
    global_perplexity_pub = nh->advertise<Perplexity>("perplexity_score", 10);
    global_surprise_pub = nh->advertise<LocalSurprise>("scene_perplexity", 10);
    local_surprise_pub = nh->advertise<LocalSurprise>("cell_perplexity", 10);
    topic_weights_pub = nh->advertise<TopicWeights>("topic_weight", 10);
    word_sub = nh->subscribe("/words", 10, &topic_model::words_callback, this);

    pose_t G{ { G_time, G_space, G_space } };
    rost = std::unique_ptr<ROST_t>(new ROST_t(static_cast<size_t>(V), static_cast<size_t>(K), k_alpha, k_beta, G));
    last_time = -1;

    if (polled_refine) { //refine when requested
        ROS_INFO("Topics will be refined on request.");
    } else { //refine automatically
        ROS_INFO("Topics will be refined online.");
        stopWork = false;
        workers = parallel_refine_online2(rost.get(), k_tau, p_refine_last_observation, static_cast<size_t>(observation_size), num_threads, &stopWork);
    }
}

topic_model::~topic_model()
{
    stopWork = true; //signal workers to stop
    for (auto t : workers) { //wait for them to stop
        t->join();
    }
}

static std::pair<std::map<pose_t, std::vector<int>>,
    std::map<pose_t, std::vector<pose_t>>>
words_for_cell_poses(WordObservation const& wordObs, int cell_size)
{
    using namespace std;
    map<pose_t, vector<int>> words_by_cell_pose;
    map<pose_t, vector<pose_t>> word_poses_by_cell_pose;

    for (size_t i = 0; i < wordObs.words.size(); ++i) {
        pose_t cell_pose, word_pose;
        cell_pose[0] = wordObs.observation_pose[0];
        word_pose[0] = wordObs.observation_pose[0];
        for (size_t d = 0; d < POSEDIM - 1; ++d) {
            cell_pose[d + 1] = wordObs.word_pose[i * (POSEDIM - 1) + d] / cell_size;
            word_pose[d + 1] = wordObs.word_pose[i * (POSEDIM - 1) + d];
        }
        words_by_cell_pose[cell_pose].push_back(wordObs.words[i]);
        word_poses_by_cell_pose[cell_pose].push_back(word_pose);
    }
    return make_pair(words_by_cell_pose, word_poses_by_cell_pose);
}

void topic_model::words_callback(const WordObservation::ConstPtr& wordObs)
{
    using namespace std;
    if (wordObs->observation_pose.empty()) {
        ROS_ERROR("Word observations are missing observation poses! Skipping...");
        return;
    }

    int observation_time = wordObs->observation_pose[0];
    //update the  list of observed time step ids
    if (observation_times.empty() || observation_times.back() < observation_time) {
        observation_times.push_back(observation_time);
    } else if (observation_times.back() > observation_time) {
        ROS_WARN("Observation received that is older than previous observation! Skipping...");
        return;
    }

    //if we are receiving observations from the next time step, then spit out
    //topics for the current time step.
    if (last_time >= 0 && (last_time != observation_time)) {
        broadcast_topics();
        size_t refine_count = rost->get_refine_count();
        ROS_INFO("#cells_refined: %u", static_cast<unsigned>(refine_count - last_refine_count));
        last_refine_count = refine_count;
        word_poses_by_cell.clear();
        current_cell_poses.clear();
    }
    last_time = observation_time;

    auto celldata = words_for_cell_poses(*wordObs, cell_width);
    auto const& cell_words = celldata.first;
    auto const& cell_word_poses = celldata.second;
    for (auto const& entry : cell_words) {
        auto const& cell_pose = entry.first;
        auto const& cell_words = entry.second;
        rost->add_observation(cell_pose, cell_words.begin(), cell_words.end(), update_topic_model);
        current_cell_poses.push_back(cell_pose);
    }
    word_poses_by_cell.insert(cell_word_poses.begin(), cell_word_poses.end());
}

void topic_model::broadcast_topics()
{
    using namespace std;
    auto const time = last_time;

    WordObservation::Ptr topicObs(new WordObservation);
    topicObs->source = "topics";
    topicObs->vocabulary_begin = 0;
    topicObs->vocabulary_size = static_cast<int32_t>(K);
    topicObs->seq = static_cast<uint32_t>(time);
    topicObs->observation_pose.push_back(time);

    Perplexity::Ptr global_perplexity(new Perplexity);
    global_perplexity->seq = static_cast<uint32_t>(time);
    global_perplexity->perplexity = -1;

    LocalSurprise::Ptr global_surprise(new LocalSurprise);
    global_surprise->seq = static_cast<uint32_t>(time);
    global_surprise->cell_width = cell_width;
    global_surprise->surprise.resize(word_poses_by_cell.size(), 0);
    global_surprise->surprise_poses.resize(word_poses_by_cell.size() * (POSEDIM - 1), 0);

    LocalSurprise::Ptr local_surprise(new LocalSurprise);
    local_surprise->seq = static_cast<uint32_t>(time);
    local_surprise->cell_width = cell_width;
    local_surprise->surprise.resize(word_poses_by_cell.size(), 0);
    local_surprise->surprise_poses.resize(word_poses_by_cell.size() * (POSEDIM - 1), 0);

    TopicWeights::Ptr msg_topic_weights(new TopicWeights);
    msg_topic_weights->seq = static_cast<uint32_t>(time);
    msg_topic_weights->weight = rost->get_topic_weights();

    auto n_words = 0ul;
    double sum_log_p_word = 0;
    auto entryIdx = 0u;

    for (auto const& entry : word_poses_by_cell) {
        const pose_t& cell_pose = entry.first;
        vector<int> topics; //topic labels for each word in the cell
        double log_likelihood; //cell's sum_w log(p(w | model) = log p(cell | model)
        tie(topics, log_likelihood) = rost->get_ml_topics_and_ppx_for_pose(cell_pose);

        auto const& cell = rost->get_cell(cell_pose);
        for (auto i = 0u; i < POSEDIM - 1; i++) {
            auto const poseIdx = entryIdx * (POSEDIM - 1) + i;
            local_surprise->surprise_poses[poseIdx] = cell_pose[i + 1];
            global_surprise->surprise_poses[poseIdx] = cell_pose[i + 1];
        }
        local_surprise->surprise[entryIdx] = rost->cell_perplexity_word(cell->W, rost->neighborhood(*cell));
        global_surprise->surprise[entryIdx] = rost->cell_perplexity_word(cell->W, rost->weight_Z);

        //populate the topic label message
        topicObs->words.insert(topicObs->words.end(), topics.begin(), topics.end());
        vector<pose_t> const& word_poses = entry.second;
        for (auto const& word_pose : word_poses) {
            for (auto i = 1u; i < POSEDIM; i++) {
                topicObs->word_pose.push_back(word_pose[i]);
            }
        }

        n_words += topics.size();
        sum_log_p_word += log_likelihood;
        assert(n_words * (POSEDIM - 1) == topicObs->word_pose.size());
        entryIdx++;
    }

    global_perplexity->perplexity = exp(-sum_log_p_word / n_words);

    scene_pub.publish(topicObs);
    topic_weights_pub.publish(msg_topic_weights);
    global_perplexity_pub.publish(global_perplexity);
    global_surprise_pub.publish(global_surprise);
    local_surprise_pub.publish(local_surprise);
}
