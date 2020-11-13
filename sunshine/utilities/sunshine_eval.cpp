#include "sunshine/common/simulation_utils.hpp"
#include "sunshine/common/metric.hpp"

using namespace sunshine;

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: ./sunshine_eval bagfile image_topic depth_topic segmentation_topic" << std::endl;
        return 1;
    }
    std::string const bagfile(argv[1]);
    std::string const image_topic_name(argv[2]);
    std::string const depth_topic_name(argv[3]);
    std::string const segmentation_topic_name(argv[4]);

    auto result = benchmark(bagfile, image_topic_name, segmentation_topic_name, depth_topic_name, sunshine::Parameters({}), nmi<4>, 10);
    std::cout << "Average NMI: " << std::get<0>(result) << std::endl;
    return 0;
}
