//
// Created by stewart on 11/12/21.
//
#ifndef SUNSHINE_PROJECT_IMAGE_PROCESSING_HPP
#define SUNSHINE_PROJECT_IMAGE_PROCESSING_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/progress.hpp>

namespace sunshine {


namespace impl {
    // Helper function to calculate the distance between 2 points.
    static double dist(cv::Point const& a, cv::Point const& b) {
        return sqrt(pow((double) (a.x - b.x), 2) + pow((double) (a.y - b.y), 2));
    }

    // Helper function that computes the longest distance from the edge to the center point.
    static double getMaxDisFromCorners(const cv::Size& imgSize, const cv::Point& center) {
        // given a rect and a line
        // get which corner of rect is farthest from the line

        std::vector<cv::Point> corners(4);
        corners[0] = cv::Point(0, 0);
        corners[1] = cv::Point(imgSize.width, 0);
        corners[2] = cv::Point(0, imgSize.height);
        corners[3] = cv::Point(imgSize.width, imgSize.height);

        double maxDis = 0;
        for (int i = 0; i < 4; ++i) {
            double dis = dist(corners[i], center);
            if (maxDis < dis) maxDis = dis;
        }

        return maxDis;
    }

    // Helper function that creates a gradient image.
    // firstPt, radius and power, are variables that control the artistic effect of the filter.
    static void generateGradient(cv::Mat& mask) {
        cv::Point firstPt = cv::Point(mask.size().width / 2, mask.size().height / 2);
        double radius     = 1.0;
        double power      = 0.5;
        double exponent   = 1.4;

        double center_factor = 0.6;
        double a             = 1 - center_factor;

        double maxImageRad   = radius * getMaxDisFromCorners(mask.size(), firstPt);
        double minBrightness = pow(cos(1. / radius * power), exponent);

        mask.setTo(cv::Scalar(1));
        for (int i = 0; i < mask.rows; i++) {
            for (int j = 0; j < mask.cols; j++) {
                double temp = dist(firstPt, cv::Point(j, i)) / maxImageRad;

                //            temp = temp * power;
                //            double temp_s = pow(cos(temp), exponent);
                //            if (minBrightness > temp_s) std::cerr << minBrightness << "," << temp_s << std::endl;
                //            mask.at<double>(i, j) = minBrightness / temp_s;

                mask.at<double>(i, j) = center_factor + a * pow(temp, exponent);
//                if (mask.at<double>(i, j) > 1) std::cerr << temp << std::endl;
            }
        }
    }

    struct ColorFilter {
        double b = 0., g = 0., r = 0., offset = 0.;

        [[nodiscard]] std::string toStr() const {
            return std::to_string(b) + "," + std::to_string(g) + "," + std::to_string(r) + "," + std::to_string(offset);
        }
    };

    struct ColorFilterMatrix {
        ColorFilter b;
        ColorFilter g;
        ColorFilter r;

        [[nodiscard]] std::string toStr() const {
            return b.toStr() + "\n" + g.toStr() + "\n" + r.toStr();
        }
    };

    template<typename T = double>
    static std::array<double, 3> hueShiftRed(std::array<T, 3> const& bgr, double const& h) {
        double const u = std::cos(h);
        double const w = std::sin(h);

        return {(0.114 - 0.114 * u - 0.497 * w) * bgr[0],
                (0.587 - 0.587 * u + 0.330 * w) * bgr[1],
                (0.299 + 0.701 * u + 0.168 * w) * bgr[2]};
    }

    static std::array<uint8_t, 2> normalizingInterval(std::vector<uint8_t> const& normalizeArray) {
        uint8_t high = 255, low = 0, maxDist = 0;
        if (normalizeArray.size() < 2) { throw std::invalid_argument("normalizeArray must contain at least 2 elements"); }
        for (auto i = 1; i < normalizeArray.size(); ++i) {
            auto dist = normalizeArray[i] - normalizeArray[i - 1];
            if (dist > maxDist) {
                maxDist = dist;
                high    = normalizeArray[i];
                low     = normalizeArray[i - 1];
            }
        }
        return {low, high};
    }

    static uint8_t clipDouble(double input) {
        return static_cast<uint8_t>(std::max(0., std::min(255., std::round(input))));
    }
}

using namespace ::sunshine::impl;

static inline cv::Mat apply_clahe(cv::Mat img) {
    cv::Mat lab_image;
    cv::cvtColor(img, lab_image, cv::COLOR_BGR2Lab);
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
    cv::Mat& image_clahe = img;
    cv::cvtColor(lab_image, image_clahe, cv::COLOR_Lab2BGR);
    return img;
}

static inline cv::Mat devignette(cv::Mat img) {
    cv::Mat maskImg(img.size(), CV_64F);
    generateGradient(maskImg);
    cv::Mat hsv_image;
    cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
    for (int row = 0; row < hsv_image.size().height; row++) {
        for (int col = 0; col < hsv_image.size().width; col++) {
            cv::Vec3b value = hsv_image.at<cv::Vec3b>(row, col);
            value.val[2] *= maskImg.at<double>(row, col);
            value.val[1] = clipDouble(value.val[1] / pow(maskImg.at<double>(row, col), 2));
            //            value.val[1] /= maskImg.at<double>(row, col);
            hsv_image.at<cv::Vec3b>(row, col) = value;
        }
    }
    cv::cvtColor(hsv_image, img, cv::COLOR_HSV2BGR);
    return img;
}



template<typename T = uint32_t>
static inline std::tuple<double, std::array<std::array<T, 256>, 3>, uint64_t> computeShiftAndHistogram(cv::Mat const& img) {
    double constexpr minAvgRed      = 60;
    double constexpr maxHueShift    = 120 * CV_PI / 180.0;

    std::tuple<double, std::array<std::array<T, 256>, 3>, uint64_t> shiftAndHists{};
    auto const avgColor = cv::mean(img);
    double avgRed       = avgColor[2];

    while (avgRed < minAvgRed) {
        std::get<0>(shiftAndHists) += CV_PI / 180.0;
        if (std::get<0>(shiftAndHists) >= maxHueShift) break;
        auto const shiftedAvg = hueShiftRed({avgColor[0], avgColor[1], avgColor[2]}, std::get<0>(shiftAndHists));
        avgRed                = shiftedAvg[0] + shiftedAvg[1] + shiftedAvg[2];
    }

    for (auto row = 0; row < img.rows; ++row) {
        for (auto col = 0; col < img.cols; ++col) {
            auto const& pixel       = img.at<cv::Vec3b>(row, col);
            auto const shiftedPixel = hueShiftRed<uint8_t>({pixel[0], pixel[1], pixel[2]}, std::get<0>(shiftAndHists));
            std::get<1>(shiftAndHists)[0][pixel[0]]++;
            std::get<1>(shiftAndHists)[1][pixel[1]]++;
            std::get<1>(shiftAndHists)[2][clipDouble(shiftedPixel[0] + shiftedPixel[1] + shiftedPixel[2])]++;
        }
    }

    std::get<2>(shiftAndHists) = img.rows * img.cols;

    return shiftAndHists;
}

template<class FileList, typename T = uint64_t>
static inline std::tuple<double, std::array<std::array<T, 256>, 3>, uint64_t> computeShiftAndHistogram(FileList const& files, bool const cache_all_images = false) {
    double constexpr minAvgRed      = 60;
    double constexpr maxHueShift    = 120 * CV_PI / 180.0;
    std::vector<cv::Mat> images;
    images.reserve(files.size());

    std::tuple<double, std::array<std::array<T, 256>, 3>, uint64_t> shiftAndHists{};
    cv::Scalar avgColor{};

    boost::progress_display bar(files.size());
    for (auto const& iter : files) {
        images.push_back(cv::imread(iter, cv::IMREAD_COLOR));
        avgColor += cv::mean(images.back());
        std::get<2>(shiftAndHists) += images.back().rows * images.back().cols;
        if (!cache_all_images) images.clear();
        ++bar;
    }
    avgColor /= static_cast<double>(files.size());

    double avgRed = avgColor[2];
    while (avgRed < minAvgRed) {
        std::get<0>(shiftAndHists) += CV_PI / 180.0;
        if (std::get<0>(shiftAndHists) >= maxHueShift) break;
        auto const shiftedAvg = hueShiftRed({avgColor[0], avgColor[1], avgColor[2]}, std::get<0>(shiftAndHists));
        avgRed                = shiftedAvg[0] + shiftedAvg[1] + shiftedAvg[2];
    }

    boost::progress_display hist_bar(files.size());
    size_t i = 0;
    for (auto const& iter : files) {
        cv::Mat const& img = (cache_all_images) ? images[i++] : cv::imread(iter, cv::IMREAD_COLOR);
        for (auto row = 0; row < img.rows; ++row) {
            for (auto col = 0; col < img.cols; ++col) {
                auto const& pixel       = img.at<cv::Vec3b>(row, col);
                auto const shiftedPixel = hueShiftRed<uint8_t>({pixel[0], pixel[1], pixel[2]}, std::get<0>(shiftAndHists));
                std::get<1>(shiftAndHists)[0][pixel[0]]++;
                std::get<1>(shiftAndHists)[1][pixel[1]]++;
                std::get<1>(shiftAndHists)[2][clipDouble(shiftedPixel[0] + shiftedPixel[1] + shiftedPixel[2])]++;
            }
        }
        ++hist_bar;
    }

    return shiftAndHists;
}

/**
     * Adapted from https://github.com/nikolajbech/underwater-image-color-correction/blob/b366f3252aebe0e76a1f824d431c94507cc5562b/index.js
 */
template<typename T = uint32_t>
static inline ColorFilterMatrix computeColorFilterMatrix(double const& hueShift, std::array<std::array<T, 256>, 3> const& histograms, double thresholdLevel = -1) {
    double const blueMagicVal   = 1.2;
    if (thresholdLevel < 0) {
        uint64_t n_pts = 0;
        for (auto const& hist : histograms) {
            for (auto const& binValue : hist) n_pts += binValue;
        }
        thresholdLevel = static_cast<double>(n_pts) / 2000.0;
    }

    std::vector<uint8_t> normalize_b(1), normalize_g(1), normalize_r(1);
    for (auto i = 0; i < 256; ++i) {
        if (histograms[0][i] < thresholdLevel) normalize_b.push_back(i);
        if (histograms[1][i] < thresholdLevel) normalize_g.push_back(i);
        if (histograms[2][i] < thresholdLevel) normalize_r.push_back(i);
    }
    normalize_b.push_back(255);
    normalize_g.push_back(255);
    normalize_r.push_back(255);

    auto const b_interval = normalizingInterval(normalize_b);
    auto const g_interval = normalizingInterval(normalize_g);
    auto const r_interval = normalizingInterval(normalize_r);
    auto const shifted    = hueShiftRed<int>({1, 1, 1}, hueShift);

    double const b_gain = 255.0 / (b_interval[1] - b_interval[0]);
    double const g_gain = 255.0 / (g_interval[1] - g_interval[0]);
    double const r_gain = 255.0 / (r_interval[1] - r_interval[0]);

    double const redBlue   = shifted[0] * r_gain * blueMagicVal;
    double const redGreen  = shifted[1] * r_gain;
    double const redAdjust = shifted[2] * r_gain;

    return {{b_gain, 0., 0., -b_gain * b_interval[0]},
            {0., g_gain, 0., -g_gain * g_interval[0]},
            {redBlue, redGreen, redAdjust, -r_gain * r_interval[0]}};
}

static inline cv::Mat color_correct(ColorFilterMatrix const& colorFilter, cv::Mat img) {
    for (auto row = 0; row < img.rows; ++row) {
        for (auto col = 0; col < img.cols; ++col) {
            auto& pixel = img.at<cv::Vec3b>(row, col);
            pixel[0] =
                clipDouble(pixel[0] * colorFilter.b.b + pixel[1] * colorFilter.b.g + pixel[2] * colorFilter.b.r + colorFilter.b.offset);
            pixel[1] =
                clipDouble(pixel[0] * colorFilter.g.b + pixel[1] * colorFilter.g.g + pixel[2] * colorFilter.g.r + colorFilter.g.offset);
            pixel[2] =
                clipDouble(pixel[0] * colorFilter.r.b + pixel[1] * colorFilter.r.g + pixel[2] * colorFilter.r.r + colorFilter.r.offset);
        }
    }
    return img;
}

static inline cv::Mat color_correct(const cv::Mat& img) {
     auto const& shiftAndHistograms = computeShiftAndHistogram(img);
     auto const& colorFilter = computeColorFilterMatrix(std::get<0>(shiftAndHistograms), std::get<1>(shiftAndHistograms), std::get<2>(shiftAndHistograms) / 2000.);
     return color_correct(colorFilter, img);
}

}

#endif // SUNSHINE_PROJECT_IMAGE_PROCESSING_HPP
