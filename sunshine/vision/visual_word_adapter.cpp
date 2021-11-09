#include "sunshine/visual_word_adapter.hpp"
#include "sunshine/image_preprocessor.hpp"

using namespace sunshine;

static cv::Mat preprocess_image(cv::Mat img, bool devignette, bool correct_colors, bool use_clahe) {
    if (devignette) img = ImagePreprocessor::devignette(img);
    if (correct_colors) img = ImagePreprocessor::color_correct(img);
    if (use_clahe) img = ImagePreprocessor::apply_clahe(img);
    return img;
}

std::unique_ptr<CategoricalObservation<int, 2, int>> VisualWordAdapter::operator()(ImageObservation const * const imgObs) {
    cv::Mat const &img = preprocess_image(imgObs->image.clone(), devignette, correct_colors, use_clahe);
    if (show_clahe) {
        cv::imshow("Original", imgObs->image);
        cv::waitKey(5);
        cv::imshow("Processed", img);
        cv::waitKey(5);
    }

    if (img_scale != 1.0) {
        cv::resize(img,
                   img,
                   cv::Size(),
                   img_scale,
                   img_scale,
                   (img_scale < 1.0)
                   ? cv::INTER_AREA
                   : cv::INTER_LINEAR);
    }

    WordObservation const z = multi_bow(img);
//    size_t const num_words = z.words.size();

    if (seq_start == 0) {
        seq_start = imgObs->timestamp;
    }
    uint32_t id = (seq_duration == 0)
                  ? imgObs->id
                  : static_cast<uint32_t>((imgObs->timestamp - seq_start) / seq_duration);
    uint32_t const &vocabulary_start = z.vocabulary_begin;
    uint32_t const &vocabulary_size = z.vocabulary_size;
    std::vector<int> const &observations = z.words;
    std::string const &frame = imgObs->frame;
    double const &timestamp = imgObs->timestamp;
    assert(z.word_pose.size() == observations.size() * 2);
    std::vector<std::array<int, 2>> observation_poses;
    for (auto i = 0ul; i < z.word_pose.size(); i += 2) observation_poses.push_back(make_array<2>(z.word_pose.begin() + i));
    return std::make_unique<CategoricalObservation<int, 2, int>>(frame,
                                                                 timestamp,
                                                                 id,
                                                                 observations,
                                                                 observation_poses,
                                                                 vocabulary_start,
                                                                 vocabulary_size);
}

std::string VisualWordAdapter::get_rost_path() const {
    std::string data_root = "/share/rost";
    char *data_root_c;
    data_root_c = getenv("ROSTPATH");
    if (data_root_c != nullptr) { //TODO: Do we still need this?
        std::cerr << "ROSTPATH: " << data_root_c << std::endl;
        data_root = data_root_c;
    }
    return data_root;
}
