#include "topic_model.hpp"

#include <rost/refinery.hpp>

using namespace sunshine;
using namespace sunshine_msgs;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "topic_model");
    ros::NodeHandle nh;

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

    ROS_INFO("Starting online topic modelling with parameters: K=%u, alpha=%f, beta=%f, gamma=%f tau=%f", K, k_alpha, k_beta, k_gamma, k_tau);

    topics_pub = nh->advertise<WordObservation>("topics", 10);
    perplexity_pub = nh->advertise<Perplexity>("perplexity", 10);
    cell_perplexity_pub = nh->advertise<LocalSurprise>("cell_perplexity", 10);
    topic_weights_pub = nh->advertise<TopicWeights>("topic_weight", 10);
    word_sub = nh->subscribe("words", 10, &topic_model::words_callback, this);

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

void topic_model::words_callback(const WordObservation::ConstPtr& words)
{
    using namespace std;
    if (words->observation_pose.empty()) {
        ROS_ERROR("Word observations are missing observation poses! Skipping...");
        return;
    }

    int observation_time = words->observation_pose[0];
    //update the  list of observed time step ids
    if (observation_times.empty() || observation_times.back() < observation_time) {
        observation_times.push_back(observation_time);
    }

    //if we are receiving observations from the next time step, then spit out
    //topics for the current time step.
    if (last_time >= 0 && (last_time != observation_time)) {
        broadcast_topics();
        size_t refine_count = rost->get_refine_count();
        ROS_INFO("#cells_refined: %u", static_cast<unsigned>(refine_count - last_refine_count));
        last_refine_count = refine_count;
        worddata_for_pose.clear();
    }
    vector<int> word_pose_cell(words->word_pose.size());

    //split the words into different cells, each with its own pose (t,x,y)
    map<pose_t, vector<int>> words_for_pose;
    for (size_t i = 0; i < words->words.size(); ++i) {
        pose_t pose{ { observation_time, words->word_pose[2 * i] / cell_width, words->word_pose[2 * i + 1] / cell_width } };
        words_for_pose[pose].push_back(words->words[i]);
        auto& v = worddata_for_pose[pose];
        v.push_back(words->word_pose[2 * i]);
        v.push_back(words->word_pose[2 * i + 1]);
        v.push_back(words->word_scale[i]);
    }

    for (auto& p : words_for_pose) {
        rost->add_observation(p.first, p.second);
        cellposes_for_time[words->seq].insert(p.first);
    }
    last_time = observation_time;
}

void topic_model::broadcast_topics()
{
    using namespace std;
    auto const time = last_time;

    //if nobody is listening, then why speak?
    if (topics_pub.getNumSubscribers() == 0 && perplexity_pub.getNumSubscribers() == 0
        && topic_weights_pub.getNumSubscribers() == 0 && cell_perplexity_pub.getNumSubscribers() == 0) {
        return;
    }

    WordObservation::Ptr z(new WordObservation);
    Perplexity::Ptr msg_ppx(new Perplexity);
    TopicWeights::Ptr msg_topic_weights(new TopicWeights);
    LocalSurprise::Ptr cell_perplexity(new LocalSurprise);
    z->source = "topics";
    z->vocabulary_begin = 0;
    z->vocabulary_size = static_cast<int32_t>(K);
    z->seq = static_cast<uint32_t>(time);
    z->observation_pose.push_back(time);
    msg_ppx->perplexity = 0;
    msg_ppx->seq = static_cast<uint32_t>(time);
    msg_topic_weights->seq = static_cast<uint32_t>(time);
    msg_topic_weights->weight = rost->get_topic_weights();
    cell_perplexity->seq = static_cast<uint32_t>(time);
    cell_perplexity->cell_width = cell_width;

    int n_words = 0;
    double sum_log_p_word = 0;

    int max_x = 0, max_y = 0;
    for (auto& pose_data : worddata_for_pose) {
        const pose_t& pose = pose_data.first;
        max_x = max(pose[1], max_x);
        max_y = max(pose[2], max_y);
    }
    cell_perplexity->surprise.resize(static_cast<size_t>((max_x + 1) * (max_y + 1)), 0);
    cell_perplexity->width = max_x + 1;
    cell_perplexity->height = max_y + 1;

    for (auto& pose_data : worddata_for_pose) {

        const pose_t& pose = pose_data.first;
        vector<int> topics; //topic labels for each word in the cell
        double log_likelihood; //cell's sum_w log(p(w | model) = log p(cell | model)
        tie(topics, log_likelihood) = rost->get_ml_topics_and_ppx_for_pose(pose_data.first);

        //populate the topic label message
        z->words.insert(z->words.end(), topics.begin(), topics.end());
        vector<int>& word_data = pose_data.second;
        assert(topics.size() * 3 == word_data.size()); //x,y,scale
        auto wi = word_data.begin();
        for (size_t i = 0; i < topics.size(); ++i) {
            z->word_pose.push_back(*wi++); //x
            z->word_pose.push_back(*wi++); //y
            z->word_scale.push_back(*wi++); //scale
        }
        n_words += topics.size();
        sum_log_p_word += log_likelihood;

        size_t idx = static_cast<size_t>(pose[2] * (max_x + 1) + pose[1]);
        cell_perplexity->surprise[idx] = static_cast<float>(exp(-log_likelihood / topics.size()));
    }

    msg_ppx->perplexity = exp(-sum_log_p_word / n_words);
    topics_pub.publish(z);
    perplexity_pub.publish(msg_ppx);
    topic_weights_pub.publish(msg_topic_weights);
    cell_perplexity_pub.publish(cell_perplexity);
}

std::pair<std::map<pose_t, std::vector<int>>,
    std::map<pose_t, std::vector<pose_t>>>
words_for_pose(WordObservation& z, int cell_size)
{
    using namespace std;
    //	std::map<pose_t, vector<int>> m;
    map<pose_t, vector<int>> out_words;
    map<pose_t, vector<pose_t>> out_poses;

    for (size_t i = 0; i < z.words.size(); ++i) {
        pose_t pose, pose_original;
        pose[0] = z.seq;
        pose_original[0] = z.seq;
        for (size_t d = 0; d < POSEDIM - 1; ++d) {
            pose[d + 1] = z.word_pose[i * (POSEDIM - 1) + d] / cell_size;
            pose_original[d + 1] = z.word_pose[i * (POSEDIM - 1) + d];
        }
        out_words[pose].push_back(z.words[i]);
        out_poses[pose].push_back(pose_original);
    }
    return make_pair(out_words, out_poses);
}
