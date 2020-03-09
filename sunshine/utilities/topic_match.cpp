/**
 * @input
 *  - A directory of timestamped topic models stored in files with name format [seconds]_[ms]_[name].bin
 *  - The vocabulary size `V` (files should contain an integer multiple of `V` 32-bit integers
 *  - The matching algorithm name (from adrost_utils.hpp)
 * @output To standard output, CSV-style data with the following attributes:
 *  - Timestamp: Line merges latest topic models up to this timestamp, one from each unique model name
 *  - Num Models: Number of topic models merged
 *  - Num Topics: Number of total topics (across all models)
 *  - Cluster Sizes: Vector of cluster sizes
 *  - Topic SSD (Euclidean): Vector of average squared euclidean distances of topics from corresponding cluster center
 *  - Topic SSD (JS): Same as above, but computed using Jensen-Shannon distance
 *  - Topic SSD (Angular): Same as above, but computed using angular distance
 *  - Topic SSD (Hellinger): Same as above, but computed using Hellinger distance
 *  - Silhouette Index (Euclidean): Vector of Silhouette index for each cluster center, computed using Euclidean distance
 *  - Silhouette Index (JS, Angular, Hellinger): As above
 */

#include <iostream>
#include <boost/filesystem.hpp>
#include <chrono>
#include <future>
//#include <boost/sort/sort.hpp>
#include "sunshine/common/csv.hpp"
#include "sunshine/common/adrost_utils.hpp"

std::vector<std::string> split_algs(const std::string &arg) {
    std::vector<std::string> algs;
    size_t next, idx = 0;
    do {
        next = arg.find(',', idx);
        algs.push_back(arg.substr(idx, next - idx));
        idx = next + 1;
    } while (next != std::string::npos);
    return algs;
}

int main(int argc, char **argv) {
    if (argc != 4) throw std::invalid_argument("Usage: <TOPIC_BIN_DIR> <VOCAB_SIZE> <MATCHING_ALGORITHM>");
    std::string const in_dir(argv[1]);
    int const V = std::stoi(argv[2]);
    auto const match_algs = split_algs(argv[3]);

    using namespace boost::filesystem;
    if (!is_directory(in_dir)) throw std::invalid_argument(in_dir + " is not a valid directory!");

    csv_writer<> writer(&std::cout);
    csv_writer<>::Row header{};
    header.append("Timestamp");
    header.append("Match Time");
    header.append("Total # of Models");
    header.append("Total # of Topics");
    header.append("Cluster Size");
    header.append("Initial SSD");
    header.append("Match Method");
    header.append("SSD L2");
    header.append("SI L2");
    header.append("DB L2");
    header.append("SSD JS");
    header.append("SI JS");
    header.append("DB JS");
//    header.append("SSD Angular");
//    header.append("SI Angular");
//    header.append("SSD Hellinger");
//    header.append("SI Hellinger");
    header.append("SSD L1");
    header.append("SI L1");
    header.append("DB L1");
    writer.write_header(header);

    std::vector<path> paths;
    std::copy(directory_iterator(in_dir), directory_iterator(), std::back_inserter(paths));
    std::sort(paths.begin(), paths.end());

    std::map<std::string, std::optional<Phi>> model_map;
    auto constexpr num_threads = 8;
    std::array<std::optional<std::future<void>>, num_threads> futures = {};
    size_t next_thread = 0;
    for (auto const &topic_bin : paths) {
        if (topic_bin.extension() != ".bin") continue;
        std::cerr << "Processing " << topic_bin.string() << " on thread " << next_thread << std::endl;

        auto const &stem = topic_bin.string().substr(topic_bin.string().find_last_of('/') + 1);
        auto const ms_idx = stem.find('_');
        auto const name_idx = (ms_idx != std::string::npos)
                              ? stem.find('_', ms_idx + 1)
                              : std::string::npos;
        if (ms_idx == std::string::npos || name_idx == std::string::npos) {
            std::cerr << "Failed to process due to invalid filename format" << std::endl;
            continue;
        }
        assert(name_idx > ms_idx && name_idx <= ms_idx + 4);
        int64_t const timestamp = std::stol(stem.substr(0, ms_idx)) * 1000000000 + std::stol(stem.substr(ms_idx + 1, name_idx)) * 1000000;
        std::string const name = std::string(stem.substr(name_idx + 1));

        std::ifstream file_reader(topic_bin.string(), std::ios::in | std::ios::binary);
        model_map[name].emplace(file_reader, name, V);
        assert(file_reader.eof());
        file_reader.close();

        std::vector<Phi> topic_models;
        topic_models.reserve(model_map.size());
        for (auto const &entry : model_map) topic_models.push_back(*(entry.second));
        auto const cur_thread = next_thread;
        do {
            if (!futures[next_thread].has_value()
                  || futures[next_thread]->wait_for(std::chrono::microseconds(100)) == std::future_status::ready) {
                break;
            }
            next_thread = (next_thread + 1) % futures.size();
            assert(next_thread >= 0 && next_thread < futures.size());
        } while (next_thread != cur_thread);
        futures[next_thread].emplace(std::async(std::launch::async, [match_algs, topic_models, timestamp, &writer]() {
            for (auto const &match_alg : match_algs) {
                auto const start = std::chrono::steady_clock::now();
                auto const results = match_topics(match_alg, topic_models);
                auto const match_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
                match_scores scores(topic_models, results.lifting);
                assert(scores.K == results.num_unique);

                csv_writer<>::Row row{};
                row.append(timestamp);
                row.append(match_duration.count());
                row.append(topic_models.size());
                row.append(results.num_unique);
                row.append(scores.cluster_sizes);
                row.append(results.ssd);
                row.append(match_alg);

                scores.compute_scores(l2_distance<double>);
                row.append(scores.mscd);
                row.append(scores.silhouette);
                row.append(scores.davies_bouldin);

                scores.compute_scores(jensen_shannon_dist<double>);
                row.append(scores.mscd);
                row.append(scores.silhouette);
                row.append(scores.davies_bouldin);

//                scores.compute_scores(angular_distance<double>);
//                row.append(scores.mscd);
//                row.append(scores.silhouette);

//                scores.compute_scores(hellinger_dist<double>);
//                row.append(scores.mscd);
//                row.append(scores.silhouette);

                scores.compute_scores(l1_distance<double>);
                row.append(scores.mscd);
                row.append(scores.silhouette);
                row.append(scores.davies_bouldin);

                writer.write_row(row);
            }
            writer.flush();
        }));
        next_thread = (next_thread + 1) % futures.size();
    }

    return 0;
}