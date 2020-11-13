//
// Created by stewart on 3/9/20.
//

#ifndef SUNSHINE_PROJECT_VISUAL_WORD_ADAPTER_HPP
#define SUNSHINE_PROJECT_VISUAL_WORD_ADAPTER_HPP

#include <visualwords/bow.hpp>
#include <visualwords/color_words.hpp>
#include <visualwords/feature_words.hpp>
#include <visualwords/texton_words.hpp>

#include "sunshine/common/observation_adapters.hpp"
#include "sunshine/common/utils.hpp"

namespace sunshine {

class VisualWordAdapter : public Adapter<VisualWordAdapter, ImageObservation, CategoricalObservation<int, 2, int>> {
    using Parent = Adapter<VisualWordAdapter, ImageObservation, CategoricalObservation<int, 2, int>>;

    bool use_clahe, show_clahe = false;
    double img_scale, seq_start = 0.0, seq_duration;
    MultiBOW multi_bow;

    [[nodiscard]] cv::Mat apply_clahe(cv::Mat img) const;
    std::string get_rost_path() const;

  public:

    template<typename ParamServer>
    explicit VisualWordAdapter(ParamServer *nh) {
        std::string const data_root = get_rost_path();

        use_clahe = nh->template param<bool>("use_clahe", false);
        show_clahe = nh->template param<bool>("show_clahe", false);
        img_scale = nh->template param<double>("scale", 1.0);
        seq_duration = nh->template param<double>("seq_duration", 0);

        bool const use_texton = nh->template param<bool>("use_texton", false);
        int const texton_cell_size = nh->template param<int>("texton_cell_size", 64);
        std::string texton_vocab_filename = nh->template param<std::string>("texton_vocab",
                                                                            data_root
                                                                                  + "/libvisualwords/data/texton.vocabulary.baraka.1000.csv");

        std::string const feature_descriptor_name = nh->template param<std::string>("feature_descriptor", "ORB");

        bool const use_surf = nh->template param<bool>("use_surf", false);
        int const num_surf = nh->template param<int>("num_surf", 1000);

        bool const use_orb = nh->template param<bool>("use_orb", true);
        int const num_orb = nh->template param<int>("num_orb", 1000);
        std::string const vocabulary_filename = nh->template param<std::string>("vocab",
                                                                                data_root + "/libvisualwords/data/orb_vocab/default.yml");

        bool const use_hue = nh->template param<bool>("use_hue", true);
        int const color_cell_size = nh->template param<int>("color_cell_size", 32);

        bool const use_intensity = nh->template param<bool>("use_intensity", true);

        std::vector<std::string> feature_detector_names;
        std::vector<int> feature_sizes;

        if (use_texton) {
            multi_bow.add(new TextonBOW(texton_cell_size, img_scale, texton_vocab_filename));
        }

        if (use_surf || use_orb) {
            if (use_surf) {
                feature_detector_names.emplace_back("SURF");
                feature_sizes.push_back(num_surf);
            }
            if (use_orb) {
                feature_detector_names.emplace_back("ORB");
                feature_sizes.push_back(num_orb);
            }
            multi_bow.add(new LabFeatureBOW(vocabulary_filename,
                                            feature_detector_names,
                                            feature_sizes,
                                            feature_descriptor_name,
                                            img_scale));
        }

        if (use_hue || use_intensity) {
            multi_bow.add(new ColorBOW(color_cell_size, img_scale, use_hue, use_intensity));
        }
    }

    typename std::unique_ptr<Parent::Output> operator()(typename Parent::Input const *imgObs);
    using Parent::operator();
};

}
#endif //SUNSHINE_PROJECT_VISUAL_WORD_ADAPTER_HPP
