#include "topic_model.hpp"

#include "utils.hpp"
#include <fstream>
#include <exception>
#include <opencv2/core.hpp>
#include <rost/refinery.hpp>
#include <sunshine_msgs/LocalSurprise.h>
#include <sunshine_msgs/Perplexity.h>
#include <sunshine_msgs/SaveObservationModel.h>
#include <sunshine_msgs/GetTopicMap.h>
#include <sunshine_msgs/TopicMap.h>
#include <sunshine_msgs/TopicWeights.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/convert.h>
#include <tf2/transform_storage.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"

using namespace sunshine;
using namespace sunshine_msgs;

static std::string const CELL_PPX = "cell";
static std::string const NEIGHBORHOOD_PPX = "neighborhood";
static std::string const GLOBAL_PPX = "global";
std::vector<std::string> const topic_model::VALID_MAP_PPX_TYPES = {CELL_PPX, NEIGHBORHOOD_PPX, /*"scene" ,*/ GLOBAL_PPX};

template<typename T>
long record_lap(T &time_checkpoint)
{
    auto const duration = std::chrono::steady_clock::now() - time_checkpoint;
    time_checkpoint = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "topic_model");
    ros::NodeHandle nh("~");

    topic_model model(&nh);

    ros::MultiThreadedSpinner spinner;
    ROS_INFO("Spinning...");
    spinner.spin();
}

topic_model::topic_model(ros::NodeHandle *nh)
    : nh(nh)
{
    std::string cell_size_string;
    bool is_hierarchical;
    int num_levels;
    double cell_size_space, cell_size_time;
    nh->param<int>("K", K, 100); // number of topics
    nh->param<int>("V", V, 1500); // vocabulary size
    nh->param<bool>("hierarchical", is_hierarchical, false);
    nh->param<int>("num_levels", num_levels, 3);
    nh->param<double>("alpha", k_alpha, 0.1);
    nh->param<double>("beta", k_beta, 1.0);
    nh->param<double>("gamma", k_gamma, 0.0001);
    nh->param<double>("tau", k_tau, 0.5); // beta(1,tau) is used to pick cells for global refinement
    nh->param<double>("p_refine_rate_local", p_refine_rate_local, 0.5); // probability of refining last observation
    nh->param<double>("p_refine_rate_global", p_refine_rate_global, 0.5);
    nh->param<int>("num_threads", num_threads, 4); // beta(1,tau) is used to pick cells for refinement
    nh->param<double>("cell_space", cell_size_space, 1);
    nh->param<std::string>("cell_size", cell_size_string, "");
    nh->param<double>("cell_time", cell_size_time, 1);
    nh->param<CellDimType>("G_time", G_time, 1);
    nh->param<CellDimType>("G_space", G_space, 1);
    nh->param<bool>("polled_refine", polled_refine, false);
    nh->param<bool>("update_topic_model", update_topic_model, true);
    nh->param<int>("min_obs_refine_time", min_obs_refine_time, 200);
    nh->param<int>("word_obs_queue_size", obs_queue_size, 1);
    nh->param<bool>("publish_topics", publish_topics, true);
    nh->param<bool>("publish_local_surprise", publish_local_surprise, true);
    nh->param<bool>("publish_global_surprise", publish_global_surprise, true);
    nh->param<bool>("publish_ppx", publish_ppx, true);
    nh->param<std::string>("words_topic", words_topic_name, "/word_extractor/words");
    nh->param<std::string>("map_ppx", map_ppx_type, "global");
    nh->param<int>("map_publish_period", map_publish_period, -1);
    if (std::find(VALID_MAP_PPX_TYPES.cbegin(), VALID_MAP_PPX_TYPES.cend(), map_ppx_type) == VALID_MAP_PPX_TYPES.cend()) {
        throw std::invalid_argument("Invalid map perplexity type: " + map_ppx_type);
    }

    if (!cell_size_string.empty()) {
        double idx = 0;
        for (size_t i = 1; i <= POSEDIM; i++) {
            auto const next = (i < POSEDIM) ? cell_size_string.find("x", idx) : cell_size_string.size();
            if (next == cell_size_string.npos) {
                throw std::invalid_argument("Cell size string '" + cell_size_string + "' is invalid!");
            }
            cell_size[i - 1] = std::stod(cell_size_string.substr(idx, next));
            idx = next + 1;
        }
    } else {
        cell_size[0] = cell_size_time;
        for (size_t i = 1; i < POSEDIM; i++) {
            cell_size[i] = cell_size_space;
        }
    }

    ROS_INFO("Starting online topic modelling with parameters: K=%u, alpha=%f, beta=%f, tau=%f", K, k_alpha, k_beta, k_tau);

    scene_pub = nh->advertise<WordObservation>("topics", 10);
    global_perplexity_pub = nh->advertise<Perplexity>("perplexity_score", 10);
    global_surprise_pub = nh->advertise<LocalSurprise>("scene_perplexity", 10);
    local_surprise_pub = nh->advertise<LocalSurprise>("cell_perplexity", 10);
    topic_weights_pub = nh->advertise<TopicWeights>("topic_weight", 10);
    map_pub = nh->advertise<TopicMap>("topic_map", 1);

    word_sub = nh->subscribe(words_topic_name, static_cast<uint32_t>(obs_queue_size), &topic_model::words_callback,
                             this);

    cell_pose_t G{{G_time, G_space, G_space, G_space}};
    if (is_hierarchical) {
        ROS_INFO("Enabling hierarchical ROST with %d levels, gamma=%f", num_levels, k_gamma);
        rost = std::make_unique<hROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t>>>(
            static_cast<size_t>(V), static_cast<size_t>(K), static_cast<size_t>(num_levels), k_alpha, k_beta, k_gamma, neighbors_t(G));

    } else {
        rost = std::make_unique<ROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t>>>(static_cast<size_t>(V), static_cast<size_t>(K),
                                                                                             k_alpha, k_beta, neighbors_t(G));
        if (k_gamma > 0) {
            ROS_INFO("Enabling HDP with gamma=%f", k_gamma);
            auto rost_concrete = dynamic_cast<ROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t>> *>(rost.get());
            rost_concrete->gamma = k_gamma;
            rost_concrete->enable_auto_topics_size(true);
        }
    }

    if (polled_refine) { //refine when requested
        throw std::runtime_error("Not implemented. Requires services.");
        ROS_INFO("Topics will be refined on request.");
    } else { //refine automatically
        ROS_INFO("Topics will be refined online.");
        stopWork = false;
        workers = parallel_refine_online_exp_beta(rost.get(), k_tau, p_refine_rate_local, p_refine_rate_global, num_threads, &stopWork);
    }

    boost::function<bool(sunshine_msgs::SaveObservationModelRequest &,
                         sunshine_msgs::SaveObservationModelResponse &)> const save_topics_by_time_csv =
        [this](sunshine_msgs::SaveObservationModelRequest &req, sunshine_msgs::SaveObservationModelResponse &) {
            std::ofstream writer(req.filename);
            writer << "time";
            for (auto k = 0; k < this->K; k++) {
                writer << ",topic_" << std::to_string(k) << "_count";
            }
            auto const topics = this->get_topics_by_time();
            for (auto const &entry : topics) {
                writer << "\n";
                writer << std::to_string(entry.first);
                for (auto const &count : entry.second) {
                    writer << "," + std::to_string(count);
                }
            }
            writer.close();
            return true;
        };

    boost::function<bool(sunshine_msgs::SaveObservationModelRequest &,
                         sunshine_msgs::SaveObservationModelResponse &)> const save_topics_by_cell_csv =
        [this](sunshine_msgs::SaveObservationModelRequest &req, sunshine_msgs::SaveObservationModelResponse &) {
            std::ofstream writer(req.filename);
            writer << "pose_dim_0";
            for (auto i = 1; i < POSEDIM; i++) {
                writer << ",pose_dim_" + std::to_string(i);
            }
            for (auto k = 0; k < this->K; k++) {
                writer << ",topic_" << std::to_string(k) << "_count";
            }
            auto const topics = this->get_topics_by_cell();
            for (auto const &entry : topics) {
                writer << "\n";
                writer << std::to_string(entry.first[0]);
                for (auto dim = 1u; dim < POSEDIM; dim++) {
                    writer << "," + std::to_string(entry.first[dim]);
                }
                for (auto const &count : entry.second) {
                    writer << "," + std::to_string(count);
                }
            }
            writer.close();
            return true;
        };

    boost::function<bool(sunshine_msgs::GetTopicSummaryRequest &, sunshine_msgs::GetTopicSummaryResponse &)> const generate_topic_summary =
        [this](GetTopicSummaryRequest &request, GetTopicSummaryResponse &response) {
            response.num_topics = this->K;
            response.last_seq = last_time;
            response.header.stamp = ros::Time::now();
            if (request.grouping == "cell") {
                auto const topics = this->get_topics_by_cell();
                response.num_observations = topics.size();
                response.pose_fields = "t,x,y,z";
                response.topic_counts.reserve(this->K * topics.size());
                response.topic_pose.reserve(POSEDIM * topics.size());
                for (auto const &entry : topics) {
                    response.topic_pose.insert(response.topic_pose.end(), entry.first.begin(), entry.first.end());
                    response.topic_counts.insert(response.topic_counts.end(), entry.second.begin(), entry.second.end());
                }
            } else if (request.grouping == "time") {
                auto const topics = this->get_topics_by_time();
                response.num_observations = topics.size();
                response.pose_fields = "t";
                response.topic_counts.reserve(this->K * topics.size());
                response.topic_pose.reserve(topics.size());
                for (auto const &entry : topics) {
                    response.topic_pose.push_back(entry.first);
                    response.topic_counts.insert(response.topic_counts.end(), entry.second.begin(), entry.second.end());
                }
            } else if (request.grouping == "global") {
                auto const topics = this->rost->get_topic_weights();
                response.num_observations = 1;
                response.pose_fields = "";
                assert(topics.size() == this->K);
                response.topic_counts.reserve(topics.size());
                response.topic_pose.reserve(0);
                for (auto const &entry : topics) {
                    response.topic_counts = topics;
                }
            } else if (request.grouping == "observation") {
                ROS_ERROR("'observation' grouping is not yet implemented.");
                return false;
            } else {
                ROS_ERROR("Unrecognized topic grouping: %s", request.grouping.c_str());
                return false;
            }
            return true;
        };

    boost::function<bool(sunshine_msgs::GetTopicMapRequest &, sunshine_msgs::GetTopicMapResponse &)> const get_topic_map =
        [this](sunshine_msgs::GetTopicMapRequest &, sunshine_msgs::GetTopicMapResponse &response) {
            response.topic_map = *generate_topic_map(last_time);
            return true;
        };

    this->time_topic_server = nh->advertiseService<>("save_topics_by_time_csv", save_topics_by_time_csv);
    this->cell_topic_server = nh->advertiseService<>("save_topics_by_cell_csv", save_topics_by_cell_csv);
    this->topic_summary_server = nh->advertiseService<>("get_topic_summary", generate_topic_summary);
    this->topic_map_server = nh->advertiseService<>("get_topic_map", get_topic_map);

    if (map_publish_period > 0) {
        map_publish_timer = nh->createTimer(ros::Duration(map_publish_period), [this](ros::TimerEvent const &) {
            map_pub.publish(*generate_topic_map(last_time));
        });
    }
}

topic_model::~topic_model()
{
    map_publish_timer.stop();
    stopWork = true; //signal workers to stop
    for (auto const &t : workers) { //wait for them to stop
        t->join();
    }
    if (broadcast_thread && broadcast_thread->joinable()) broadcast_thread->join();
}

std::map<ROST_t::pose_dim_t, std::vector<int>> topic_model::get_topics_by_time() const
{
    auto const poses_by_time = rost->get_poses_by_time();
    std::map<ROST_t::pose_dim_t, std::vector<int>> topics_by_time;
    for (auto const &entry : poses_by_time) {
        std::vector<int> topics(static_cast<size_t>(K), 0);
        for (auto const &pose : entry.second) {
            for (auto const &topic : rost->get_topics_for_pose(pose)) {
                topics[static_cast<size_t>(topic)] += 1;
            }
        }
        topics_by_time.insert({entry.first, topics});
    }
    return topics_by_time;
}

std::map<cell_pose_t, std::vector<int>> topic_model::get_topics_by_cell() const
{
    auto const &poses = rost->cell_pose;
    std::map<cell_pose_t, std::vector<int>> topics_by_cell;
    for (auto const &pose : poses) {
        auto const cell = rost->get_cell(pose);
        if (cell->nZ.size() == this->K) {
            topics_by_cell.insert({pose, cell->nZ});
        } else {
            ROS_WARN_THROTTLE(1, "topic_model::get_topics_by_cell() : Skipping cells with wrong number of topics");
            continue;
        }
    }
    return topics_by_cell;
}

static std::map<cell_pose_t, std::vector<int>>
words_for_cell_poses(WordObservation const &wordObs, std::array<double, POSEDIM> cell_size)
{
    using namespace std;
    map<cell_pose_t, vector<int>> words_by_cell_pose;

    for (size_t i = 0; i < wordObs.words.size(); ++i) {
        geometry_msgs::Point word_point;
        transformPose(word_point, wordObs.word_pose, i * 3, wordObs.observation_transform);

        word_pose_t word_stamped_point;
        word_stamped_point[0] = static_cast<WordDimType>(wordObs.observation_transform.header.stamp.toSec()); // TODO (Stewart Jamieson): Use the correct time
        word_stamped_point[1] = static_cast<WordDimType>(word_point.x);
        word_stamped_point[2] = static_cast<WordDimType>(word_point.y);
        word_stamped_point[3] = static_cast<WordDimType>(word_point.z);
        cell_pose_t cell_stamped_point = toCellPose(word_stamped_point, cell_size);

        words_by_cell_pose[cell_stamped_point].push_back(wordObs.words[i]);
    }
    return words_by_cell_pose;
}

void topic_model::wait_for_processing() const
{
    using namespace std::chrono;
    auto const elapsedSinceAdd = steady_clock::now() - lastWordsAdded;
    ROS_DEBUG("Time elapsed since last observation added (minimum set to %d ms): %lu ms",
              min_obs_refine_time, duration_cast<milliseconds>(elapsedSinceAdd).count());
    if (duration_cast<milliseconds>(elapsedSinceAdd).count() < min_obs_refine_time) {
        ROS_WARN("New word observation received too soon! Delaying...");
        if (++consecutive_rate_violations > obs_queue_size) {
            ROS_ERROR("Next word observation will likely be dropped. Increase queue size, or reduce observation rate or processing time.");
        }
        std::this_thread::sleep_for(milliseconds(min_obs_refine_time) - elapsedSinceAdd);
    } else {
        consecutive_rate_violations = 0;
    }
}

void topic_model::words_callback(const WordObservation::ConstPtr &wordObs)
{
    auto time_checkpoint = std::chrono::steady_clock::now();
    auto const time_start = time_checkpoint;

    using namespace std;
    lock_guard<mutex> guard(wordsReceivedLock);
    auto duration_lock = record_lap(time_checkpoint);

    // TODO Can observation transform ever be invalid?

    int observation_time = int(wordObs->seq);
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
        ROS_DEBUG("Received newer word observations - broadcasting observations for time %d", last_time);
        if (broadcast_thread && broadcast_thread->joinable()) {
            broadcast_thread->join();
        }
        broadcast_thread = std::make_shared<std::thread>(&topic_model::broadcast_topics, this, last_time, std::move(current_cell_poses));
        size_t refine_count = rost->get_refine_count();
        ROS_DEBUG("#cells_refined: %u", static_cast<unsigned>(refine_count - last_refine_count));
        last_refine_count = refine_count;
        current_cell_poses.clear();
        current_source.clear();
    }
    last_time = observation_time;
    auto const duration_broadcast = record_lap(time_checkpoint);

    ROS_DEBUG("Adding %lu word observations from time %d", wordObs->words.size(), observation_time);
    ROS_ERROR_COND(!current_source.empty() && current_source != wordObs->source,
                   "Words received from different source with same observation time!");
    {
        std::lock_guard<std::mutex> guard(rostLock);
        duration_lock += record_lap(time_checkpoint);

        auto const &words_by_cell_pose = words_for_cell_poses(*wordObs, cell_size);
        current_cell_poses.reserve(current_cell_poses.size() + words_by_cell_pose.size());
        for (auto const &entry : words_by_cell_pose) {
            auto const &cell_pose = entry.first;
            auto const &cell_words = entry.second;
            rost->add_observation(cell_pose, cell_words.begin(), cell_words.end(), update_topic_model);
            current_cell_poses.push_back(cell_pose);
            current_source = wordObs->source;
        }
    }
    auto const duration_add_observations = record_lap(time_checkpoint);
    ROS_DEBUG("Refining %lu cells", current_cell_poses.size());

    auto const total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - time_start).count();
    if (total_duration > this->min_obs_refine_time) {
        ROS_WARN("Words observation overhead: %lu ms (%lu lock, %lu broadcast, %lu updating cells) exceeds min refine time %d",
                 total_duration, duration_lock, duration_broadcast, duration_add_observations, min_obs_refine_time);
    } else {
        ROS_INFO("Words observation overhead: %lu ms (%lu lock, %lu broadcast, %lu updating cells)", total_duration, duration_lock,
                 duration_broadcast, duration_add_observations);
    }

    lastWordsAdded = chrono::steady_clock::now();
}

TopicMapPtr topic_model::generate_topic_map(int const obs_time) const
{
    TopicMap::Ptr topic_map(new TopicMap);
    topic_map->seq = static_cast<uint32_t>(obs_time);
    topic_map->vocabulary_begin = 0;
    topic_map->vocabulary_size = static_cast<int32_t>(K);
    topic_map->ppx_type = map_ppx_type;
    topic_map->cell_topics.reserve(rost->cell_pose.size());
    topic_map->cell_ppx.reserve(rost->cell_pose.size());
    topic_map->cell_poses.reserve(rost->cell_pose.size() * (POSEDIM - 1));
    topic_map->cell_width = {cell_size.begin() + 1, cell_size.end()};
    topic_map->observation_transform.transform.rotation.w = 1; // Identity rotation (global frame)
    topic_map->observation_transform.header.stamp = ros::Time::now();

    auto const poses = rost->cell_pose;
    auto const BATCH_SIZE = 500ul;
    auto i = 0ul;
    while (i < poses.size()) {
        std::lock_guard<std::mutex> guard(rostLock);
        auto const BATCH_END = std::min(i + BATCH_SIZE, poses.size());
        for (; i < BATCH_END; ++i) {
            auto const &cell_pose = poses[i];
            auto const &cell = rost->get_cell(cell_pose);
            auto const word_pose = toWordPose(cell_pose, cell_size);
            auto const ml_cell_topic = std::max_element(cell->nZ.cbegin(), cell->nZ.cend());
            if (ml_cell_topic == cell->nZ.cend()) {
                ROS_ERROR("Cell has no topics! Map will contain invalid topic labels.");
                continue;
            } else {
                topic_map->cell_topics.push_back(int32_t(ml_cell_topic - cell->nZ.cbegin()));
            }
            for (size_t i = 1; i < POSEDIM; i++) {
                topic_map->cell_poses.push_back(word_pose[i]);
            }
            auto const map_ppx = (map_ppx_type == NEIGHBORHOOD_PPX) ? rost->cell_perplexity_word(cell->W, rost->neighborhood(*cell))
                                                                    : rost->cell_perplexity_word(cell->W, rost->get_topic_weights());
            topic_map->cell_ppx.push_back(map_ppx);
        }
    }
    return topic_map;
}

void topic_model::broadcast_topics(int const obs_time, std::vector<cell_pose_t> broadcast_poses) const
{
    if (!publish_global_surprise && !publish_local_surprise && !publish_ppx && !publish_topics) {
        return;
    }

    auto time_checkpoint = std::chrono::steady_clock::now();
    auto const time_start = time_checkpoint;

    using namespace std;
    wait_for_processing();
    auto const duration_wait = record_lap(time_checkpoint);

    WordObservation::Ptr topicObs(new WordObservation);
    topicObs->source = current_source;
    topicObs->vocabulary_begin = 0;
    topicObs->vocabulary_size = static_cast<int32_t>(K);
    topicObs->seq = static_cast<uint32_t>(obs_time);
    topicObs->observation_transform.transform.rotation.w = 1; // Identity rotation (global frame)
    topicObs->observation_transform.header.stamp = ros::Time::now();

    Perplexity::Ptr global_perplexity(new Perplexity);
    global_perplexity->seq = static_cast<uint32_t>(obs_time);
    global_perplexity->perplexity = -1;

    LocalSurprise::Ptr global_surprise(new LocalSurprise);
    global_surprise->seq = static_cast<uint32_t>(obs_time);
    global_surprise->cell_width = {cell_size.begin() + 1, cell_size.end()};
    global_surprise->surprise.reserve(current_cell_poses.size());
    global_surprise->surprise_poses.reserve(current_cell_poses.size() * (POSEDIM - 1));

    LocalSurprise::Ptr local_surprise(new LocalSurprise);
    local_surprise->seq = static_cast<uint32_t>(obs_time);
    local_surprise->cell_width = {cell_size.begin() + 1, cell_size.end()};
    local_surprise->surprise.reserve(current_cell_poses.size());
    local_surprise->surprise_poses.reserve(current_cell_poses.size() * (POSEDIM - 1));

    TopicWeights::Ptr msg_topic_weights(new TopicWeights);
    msg_topic_weights->seq = static_cast<uint32_t>(obs_time);
    msg_topic_weights->weight = rost->get_topic_weights();

    auto n_words = 0ul;
    double sum_log_p_word = 0;
    auto entryIdx = 0u;

    bool const topics_required = publish_topics;
    bool const cell_ppx_required = publish_ppx;
    bool const neighborhood_ppx_required = publish_local_surprise;
    bool const global_ppx_required = publish_global_surprise;

    auto const duration_init = record_lap(time_checkpoint);
    long duration_lock;
    {
        std::lock_guard<std::mutex> guard(rostLock);
        duration_lock = record_lap(time_checkpoint);
        for (auto const &cell_pose : broadcast_poses) {
            auto const &cell = rost->get_cell(cell_pose);
            auto const word_pose = toWordPose(cell_pose, cell_size);

            vector<int> topics; //topic labels for each word in the cell
            double cell_log_likelihood = 0; //cell's sum_w log(p(w | model) = log p(cell | model)
            double cell_ppx, neighborhood_ppx, global_ppx;

            if (topics_required || cell_ppx_required) {
                tie(topics, cell_log_likelihood) = rost->get_ml_topics_and_ppx_for_pose(cell_pose);

                if (publish_topics) {
                    //populate the topic label message
                    topicObs->words.reserve(topicObs->words.size() + topics.size());
                    topicObs->word_pose.reserve(topicObs->word_pose.size() + topics.size() * POSEDIM);
                    topicObs->words.insert(topicObs->words.end(), topics.begin(), topics.end());
                    for (size_t i = 0; i < topics.size(); i++) {
                        for (size_t j = 1; j < POSEDIM; j++) {
                            topicObs->word_pose.push_back(word_pose[j]);
                        }
                    }
                }

                cell_ppx = exp(-cell_log_likelihood / topics.size());
                n_words += topics.size(); // used to compute total ppx
                assert(n_words * (POSEDIM - 1) == topicObs->word_pose.size());
            }

            if (neighborhood_ppx_required) {
                neighborhood_ppx = rost->cell_perplexity_word(cell->W, rost->neighborhood(*cell));
            }

            if (global_ppx_required) {
                global_ppx = rost->cell_perplexity_word(cell->W, rost->get_topic_weights());
            }

            if (publish_local_surprise || publish_global_surprise) {
                for (size_t i = 1; i < POSEDIM; i++) {
                    local_surprise->surprise_poses.push_back(cell_pose[i]);
                    global_surprise->surprise_poses.push_back(cell_pose[i]);
                }
                local_surprise->surprise.push_back(neighborhood_ppx);
                global_surprise->surprise.push_back(global_ppx);
            }

            sum_log_p_word += cell_log_likelihood;
            entryIdx++;
        }
    }
    auto const duration_populate = record_lap(time_checkpoint);

    if (publish_topics) {
        scene_pub.publish(topicObs);
        topic_weights_pub.publish(msg_topic_weights);
    }

    if (publish_ppx) {
        global_perplexity->perplexity = exp(-sum_log_p_word / n_words);
        ROS_INFO("Perplexity: %f", global_perplexity->perplexity);
        global_perplexity_pub.publish(global_perplexity);
    }

    if (publish_global_surprise) {
        global_surprise_pub.publish(global_surprise);
    }

    if (publish_local_surprise) {
        local_surprise_pub.publish(local_surprise);
    }
    auto const duration_publish = record_lap(time_checkpoint);

    if (map_publish_period == 0) {
        auto topic_map = generate_topic_map(obs_time);
        map_pub.publish(topic_map);
    }
    auto const duration_map = record_lap(time_checkpoint);

    auto const total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - time_start).count();
    ROS_INFO("Broadcast overhead: %lu ms (%lu waiting, %lu initializing, %lu locked, %lu populating messages, %lu publishing, %lu mapping)",
             total_duration, duration_wait, duration_init, duration_lock, duration_populate, duration_publish, duration_map);
}
