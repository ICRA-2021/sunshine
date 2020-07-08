//
// Created by stewart on 3/9/20.
//

#include "word_extractor_node.hpp"
#include "sunshine/visual_word_adapter.hpp"

int main(int argc, char **argv) {
    // Setup ROS node
    ros::init(argc, argv, "word_extractor");
    ros::NodeHandle nh("~");

    sunshine::WordExtractorNode<sunshine::VisualWordAdapter> visualWordExtractor(&nh);
    visualWordExtractor.spin();

    return 0;
}
