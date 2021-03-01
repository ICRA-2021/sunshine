//
// Created by stewart on 3/9/20.
//

#include "word_extractor_node.hpp"
#include "sunshine/visual_word_adapter.hpp"
#include <chrono>
#include <thread>

int main(int argc, char **argv) {
//    std::this_thread::sleep_for(std::chrono::seconds(8));
    // Setup ROS node
    ros::init(argc, argv, "word_extractor");
    ros::NodeHandle nh("~");

    sunshine::WordExtractorNode<sunshine::VisualWordAdapter> visualWordExtractor(&nh);
    visualWordExtractor.spin();

    return 0;
}
