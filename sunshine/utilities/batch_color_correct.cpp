//
// Created by stewart on November 23, 2021.
//

#include <boost/progress.hpp>
#include "sunshine/common/image_processing.hpp"
#include <filesystem>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    using namespace std::filesystem;
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory>" << std::endl;
    }

    std::cout << "Reading in files..." << std::endl;
    std::vector<std::string> files;
    for (auto const& entry : directory_iterator(argv[1])) {
        if (!entry.is_regular_file()) continue;
        files.push_back(entry.path());
    }
    std::cout << "Found " << files.size() << " files." << std::endl;
    path const dir_path{argv[2]};

    std::cout << "Computing correction matrix..." << std::endl;
    auto const shiftAndHistograms = sunshine::computeShiftAndHistogram(files, true);
    auto const colorFilter = sunshine::computeColorFilterMatrix(std::get<0>(shiftAndHistograms), std::get<1>(shiftAndHistograms), std::get<2>(shiftAndHistograms) / 2000.);
    std::cout << "Computed correction matrix." << std::endl;

    std::cout << "Processing images..." << std::endl;
    boost::progress_display bar(files.size());

    std::vector<std::thread> workers;
    unsigned long const batch_size = std::max(1ul,  files.size() / std::thread::hardware_concurrency());
    size_t start = 0;
    while (start < files.size()) {
        size_t const end = std::min(files.size(), start + batch_size);
        workers.emplace_back([&, start, end](){
            for (auto i = start; i < end; ++i) {
                cv::Mat const img = sunshine::color_correct(colorFilter, cv::imread(files[i], cv::IMREAD_COLOR));
                cv::imwrite(dir_path / basename(files[i].c_str()), img);
                ++bar;
            }
        });
        start += batch_size;
    }
    for (auto& worker : workers) if (worker.joinable()) worker.join();
    std::cout << "Finished." << std::endl;
}