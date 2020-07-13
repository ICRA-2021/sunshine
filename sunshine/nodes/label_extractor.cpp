//
// Created by stewart on 3/9/20.
//

#include "word_extractor_node.hpp"
#include "sunshine/semantic_label_adapter.hpp"

int main(int argc, char **argv) {
    // Setup ROS node
    ros::init(argc, argv, "label_extractor");
    ros::NodeHandle nh("~");

    sunshine::WordExtractorNode<sunshine::SemanticLabelAdapter<int>> labelExtractor(&nh);
    labelExtractor.spin();

    return 0;
}
