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
#include <sunshine/common/data_proc_utils.hpp>
//#include <boost/sort/sort.hpp>
#include "sunshine/common/csv.hpp"
#include "sunshine/common/matching_utils.hpp"

using namespace sunshine;

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
    if (argc != 3) throw std::invalid_argument("Usage: <TOPIC_BIN_DIR> <MATCHING_ALGORITHM>");
    std::string const in_dir(argv[1]);
    auto const match_algs = split_algs(argv[2]);

    using namespace boost::filesystem;
    if (!is_directory(in_dir)) throw std::invalid_argument(in_dir + " is not a valid directory!");

    std::vector<path> paths;
    std::copy(directory_iterator(in_dir), directory_iterator(), std::back_inserter(paths));
    std::sort(paths.begin(), paths.end());

    std::vector<std::string> topic_model_paths;
    std::vector<Phi> topic_models;
    for (auto const &topic_bin : paths) {
        if (topic_bin.extension() != ".bin") continue;
        std::cerr << "Processing " << topic_bin.string() << std::endl;
//        auto const &stem = topic_bin.string().substr(topic_bin.string().find_last_of('/') + 1);
        CompressedFileReader reader(topic_bin.string());
        topic_models.emplace_back(reader.read<Phi>());
        topic_models[topic_models.size() - 1].validate(true);
        topic_model_paths.emplace_back(topic_bin.string());

        assert(reader.eof());
    }

//    csv_writer<> scores_writer((path(in_dir) / path("match_scores.csv")).string());
//    csv_writer<>::Row scores_header{};
//    header.append("Timestamp");
//    header.append("Match Time");
//    header.append("Total # of Models");
//    header.append("Total # of Topics");
//    header.append("Cluster Size");
//    header.append("Initial SSD");
//    header.append("Match Method");
//    header.append("SSD L2");
//    header.append("SI L2");
//    header.append("SSD JS");
//    header.append("SI JS");
//    header.append("SSD Angular");
//    header.append("SI Angular");
//    header.append("SSD Hellinger");
//    header.append("SI Hellinger");
//    header.append("SSD L1");
//    header.append("SI L1");
//    writer.write_header(header);
//    scores_writer.write_header(scores_header);

    for (auto const &match_alg : match_algs) {
        auto const start = std::chrono::steady_clock::now();
        auto const results = match_topics(match_alg, topic_models);
        auto const match_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
        assert(results.lifting.size() == topic_models.size());

        csv_writer<> writer((path(in_dir) / path(match_alg + "_lifting.csv")).string());
        csv_writer<>::Row header{};
        header.append("file");
        header.append("lifting");
        writer.write_header(header);

        for (auto i = 0; i < topic_models.size(); ++i) {
            csv_writer<>::Row row{};

            row.append(topic_model_paths[i]);
            row.append(results.lifting[i]);

            writer.write_row(row);
        }
        writer.close();

//        match_scores scores(topic_models, results.lifting);
//        assert(scores.K == results.num_unique);
//
//        csv_writer<>::Row row{};
//        row.append(timestamp);
//        row.append(match_duration.count());
//        row.append(topic_models.size());
//        row.append(results.num_unique);
//        row.append(scores.cluster_sizes);
//        row.append(results.ssd);
//        row.append(match_alg);
//
//        scores.compute_scores(l2_distance<double>);
//        row.append(scores.mscd);
//        row.append(scores.silhouette);
//
//        scores.compute_scores(jensen_shannon_dist<double>);
//        row.append(scores.mscd);
//        row.append(scores.silhouette);
//
//        scores.compute_scores(angular_distance<double>);
//        row.append(scores.mscd);
//        row.append(scores.silhouette);
//
//        scores.compute_scores(hellinger_dist<double>);
//        row.append(scores.mscd);
//        row.append(scores.silhouette);
//
//        scores.compute_scores(l1_distance<double>);
//        row.append(scores.mscd);
//        row.append(scores.silhouette);
//
//        writer.write_row(row);
    }

    return 0;
}