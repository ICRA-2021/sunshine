#include <opencv2/core/types_c.h>
#include "../include/sunshine/visual_word_adapter.hpp"

using namespace sunshine;


// Helper function to calculate the distance between 2 points.
double dist(cv::Point a, cv::Point b)
{
    return sqrt(pow((double) (a.x - b.x), 2) + pow((double) (a.y - b.y), 2));
}

// Helper function that computes the longest distance from the edge to the center point.
double getMaxDisFromCorners(const cv::Size& imgSize, const cv::Point& center)
{
    // given a rect and a line
    // get which corner of rect is farthest from the line

    std::vector<cv::Point> corners(4);
    corners[0] = cv::Point(0, 0);
    corners[1] = cv::Point(imgSize.width, 0);
    corners[2] = cv::Point(0, imgSize.height);
    corners[3] = cv::Point(imgSize.width, imgSize.height);

    double maxDis = 0;
    for (int i = 0; i < 4; ++i)
    {
        double dis = dist(corners[i], center);
        if (maxDis < dis)
            maxDis = dis;
    }

    return maxDis;
}

// Helper function that creates a gradient image.
// firstPt, radius and power, are variables that control the artistic effect of the filter.
void generateGradient(cv::Mat& mask)
{
    cv::Point firstPt = cv::Point(mask.size().width/2, mask.size().height/2);
    double radius = 1.0;
    double power = 0.5;
    double exponent = 1.4;

    double center_factor = 0.6;
    double a = 1 - center_factor;

    double maxImageRad = radius * getMaxDisFromCorners(mask.size(), firstPt);
    double minBrightness = pow(cos(1. / radius * power), exponent);

    mask.setTo(cv::Scalar(1));
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            double temp = dist(firstPt, cv::Point(j, i)) / maxImageRad;

//            temp = temp * power;
//            double temp_s = pow(cos(temp), exponent);
//            if (minBrightness > temp_s) std::cerr << minBrightness << "," << temp_s << std::endl;
//            mask.at<double>(i, j) = minBrightness / temp_s;

            mask.at<double>(i, j) = center_factor + a * pow(temp, exponent);
            if (mask.at<double>(i, j) > 1) std::cerr << temp << std::endl;
        }
    }
}

cv::Mat VisualWordAdapter::apply_clahe(cv::Mat img) const {
    // Adapted from https://stackoverflow.com/a/24341809
    cv::Mat &lab_image = img;
    if (show_clahe) {
        cv::imshow("Original", img);
        cv::waitKey(5);
    }
//    cv::cvtColor(img, lab_image, cv::COLOR_BGR2Lab);

    cv::Mat maskImg(img.size(), CV_64F);
    generateGradient(maskImg);
    if (show_clahe) {
        cv::imshow("Mask", maskImg);
        cv::waitKey(5);
    }
    cv::Mat hsv_image;
    cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
    for (int row = 0; row < hsv_image.size().height; row++)
    {
        for (int col = 0; col < hsv_image.size().width; col++)
        {
            cv::Vec3b value = hsv_image.at<cv::Vec3b>(row, col);
            value.val[2] *= maskImg.at<double>(row, col);
            value.val[1] /= pow(maskImg.at<double>(row, col), 2);
//            value.val[1] /= maskImg.at<double>(row, col);
            hsv_image.at<cv::Vec3b>(row, col) =  value;
        }
    }
    cv::Mat devignette;
    cv::cvtColor(hsv_image, devignette, cv::COLOR_HSV2BGR);
    if (show_clahe) {
        cv::imshow("De-Vignetted", devignette);
        cv::waitKey(5);
    }
    cv::cvtColor(devignette, lab_image, cv::COLOR_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes); // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2);
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    // convert back to RGB
    cv::Mat &image_clahe = img;
    cv::cvtColor(lab_image, image_clahe, cv::COLOR_Lab2BGR);
    if (show_clahe) {
        cv::imshow("CLAHE", image_clahe);
        cv::waitKey(5);
    }
    return img;
}

std::unique_ptr<CategoricalObservation<int, 2, int>> VisualWordAdapter::operator()(ImageObservation const * const imgObs) {
    cv::Mat const &img = (use_clahe)
                         ? apply_clahe(imgObs->image)
                         : imgObs->image;

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
