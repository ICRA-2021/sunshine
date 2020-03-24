#include "../include/sunshine/visual_word_adapter.hpp"

using namespace sunshine;

cv::Mat VisualWordAdapter::apply_clahe(cv::Mat img) const {
    // Adapted from https://stackoverflow.com/a/24341809
    cv::Mat &lab_image = img;
    cv::cvtColor(img, lab_image, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes); // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    // convert back to RGB
    cv::Mat &image_clahe = img;
    cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
    if (show_clahe) {
        cv::imshow("CLAHE", image_clahe);
        cv::waitKey(5);
    }
    return img;
}

CategoricalObservation<int, 2, int> VisualWordAdapter::operator()(ImageObservation const &imgObs) {
    cv::Mat const &img = (use_clahe)
                         ? apply_clahe(imgObs.image)
                         : imgObs.image;

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
    size_t const num_words = z.words.size();

    if (seq_start == 0) {
        seq_start = imgObs.timestamp;
    }
    uint32_t id = (seq_duration == 0)
                  ? imgObs.id
                  : static_cast<uint32_t>((imgObs.timestamp - seq_start) / seq_duration);
    uint32_t const &vocabulary_start = z.vocabulary_begin;
    uint32_t const &vocabulary_size = z.vocabulary_size;
    std::vector<int> const &observations = z.words;
    std::string const &frame = imgObs.frame;
    double const &timestamp = imgObs.timestamp;
    assert(z.word_pose.size() == observations.size() * 2);
    std::vector<std::array<int, 2>> observation_poses;
    for (auto i = 0ul; i < z.word_pose.size(); i += 2) observation_poses.push_back(make_array<2>(z.word_pose.begin() + i));
    return CategoricalObservation<int, 2, int>(imgObs.frame,
                                               imgObs.timestamp,
                                               imgObs.id,
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
