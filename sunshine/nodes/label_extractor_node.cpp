//
// Created by stewart on 3/9/20.
//

#include "word_extractor_node.hpp"

int main(int argc, char **argv) {
    // Setup ROS node
    ros::init(argc, argv, "label_extractor");
    ros::NodeHandle nh("~");

    // TODO use label extractor
//    sunshine::WordExtractorNode<VisualWordAdapter> visualWordExtractor(&nh);
//    visualWordExtractor.spin();

    return 0;
}
