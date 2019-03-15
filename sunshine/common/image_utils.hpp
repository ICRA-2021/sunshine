#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include "utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <exception>
#include <sensor_msgs/PointCloud2.h>

namespace sunshine {
typedef double_t DepthType;

cv::Mat scaleImage(cv::Mat const& image, double const& scale, uint32_t const& interpolation = cv::INTER_CUBIC)
{
    assert(scale > 0);
    cv::Mat scaled(std::ceil(image.rows * scale), std::ceil(image.cols * scale), image.type());
    cv::resize(image, scaled, scaled.size(), 0, 0, interpolation);
    return scaled;
}

class ImageScanner {
protected:
    cv::Mat const image;
    double const scale_factor;
    double const half_frame_width, half_frame_height;
    double x, y;
private:

    bool isValidPosition() const
    {
        uint32_t const lx = std::lround((x - half_frame_width) / scale_factor), ly = std::lround((y - half_frame_height) / scale_factor);
        uint32_t const scaled_width = std::lround(getFrameWidth() / scale_factor), scaled_height = std::lround(getFrameHeight() / scale_factor);
        return lx >= 0 && (lx + scaled_width) < image.cols && ly >= 0 && (ly + scaled_height) < image.rows;
    }

    cv::Rect getViewRect() const
    {
        if (!isValidPosition()) {
            uint32_t const lx = std::lround((x - half_frame_width) / scale_factor), ly = std::lround((y - half_frame_height) / scale_factor);
            uint32_t const scaled_width = std::lround(getFrameWidth() / scale_factor), scaled_height = std::lround(getFrameHeight() / scale_factor);
            throw std::domain_error("Cannot create valid view from position " + std::to_string(x) + "," + std::to_string(y));
        }
        cv::Rect roi;
        roi.x = static_cast<int>(std::lround((x - half_frame_width) / scale_factor));
        roi.y = static_cast<int>(std::lround((y - half_frame_height) / scale_factor));
        roi.width = getFrameWidth() / scale_factor;
        roi.height = getFrameHeight() / scale_factor;
        std::cout << roi.x << roi.y << roi.width << roi.height << std::endl;
        return roi;
    }

protected:

    uint32_t getFrameWidth() const
    {
        return static_cast<uint32_t>(std::lround(half_frame_width * 2));
    }

    uint32_t getFrameHeight() const
    {
        return static_cast<uint32_t>(std::lround(half_frame_height * 2));
    }

    cv::Mat getScaledImageView(cv::Mat const& image) const {
        return scaleImage(image(getViewRect()), scale_factor);
    }

public:
    ImageScanner(cv::Mat image, uint32_t frame_width, uint32_t frame_height, double scale = 1., double start_x = 0, double start_y = 0)
        : image(image)
        , scale_factor(scale)
        , half_frame_width(frame_width / 2.)
        , half_frame_height(frame_height / 2.)
        , x((start_x > 0) ? start_x : half_frame_width)
        , y((start_y > 0) ? start_y : half_frame_height)
    {
    }

    virtual ~ImageScanner() = default;

    double getScaleFactor() const
    {
        return scale_factor;
    }

    cv::Mat getCurrentView() const
    {
        return getScaledImageView(image);
    }

    virtual void moveTo(double const x, double const y)
    {
        this->x = x;
        this->y = y;
    }

    void move(double const dx, double const dy)
    {
        this->moveTo(this->x + dx, this->y + dy);
    }

    virtual double getMinX() const {
        return half_frame_width;
    }

    virtual double getMinY() const {
        return half_frame_height;
    }

    virtual double getMaxX() const {
        return image.cols * scale_factor - 1;
    }

    virtual double getMaxY() const {
        return image.rows * scale_factor - 1;
    }

    virtual double getX() const {
        return x;
    }

    virtual double getY() const {
        return y;
    }
};

union RGBPointCloudElement {
    uint8_t bytes[sizeof(float) * 3 + sizeof(uint32_t)]; // to enforce size
    struct {
        float x;
        float y;
        float z;
        uint8_t rgb[3];
    } data;
};

template<typename DepthType = double>
cv::Mat getFlatHeightMap(uint32_t width, uint32_t height, DepthType z) {
    return cv::Mat(height, width, cvType<DepthType>::value, cv::Scalar(z));
}

template <typename DepthType = double>
class ImageScanner3D : public ImageScanner {
protected:
    cv::Mat const heightMap;
    DepthType const z;
    double const pixel_scale;
    sensor_msgs::PointCloud2Ptr pc;

public:

    virtual void moveTo(double x, double y) override {
        this->x = x / pixel_scale;
        this->y = y / pixel_scale;
    }

    virtual double getMinX() const override {
        return half_frame_width * pixel_scale;
    }

    virtual double getMinY() const override {
        return half_frame_height * pixel_scale;
    }

    virtual double getMaxX() const override {
        return (image.cols * scale_factor - 1) * pixel_scale;
    }

    virtual double getMaxY() const override {
        return (image.rows * scale_factor - 1) * pixel_scale;
    }

    virtual double getX() const override {
        return x * pixel_scale;
    }

    virtual double getY() const override {
        return y * pixel_scale;
    }

    ImageScanner3D(cv::Mat image, uint32_t frame_width, uint32_t frame_height, cv::Mat heightMap, double scale = 1., double start_z = 1., double pixel_scale = 1)
        : ImageScanner(image, frame_width, frame_height, scale, (frame_width / 2.), (frame_height / 2.))
        , heightMap((scale == 1.) ? heightMap : scaleImage(heightMap, scale))
        , z(start_z)
        , pixel_scale(pixel_scale)
        , pc(createPointCloud(getFrameWidth(), getFrameHeight(), "rgb"))
    {
        assert(heightMap.type() == sunshine::cvType<DepthType>::value);
    }

    cv::Mat getCurrentDepthView() const
    {
        cv::Mat depthMap = this->getScaledImageView(heightMap);
        for (auto row = 0; row < getFrameHeight(); row++) {
            for (auto col = 0; col < getFrameWidth(); col++) {
                assert(z >= heightMap.at<DepthType>(row, col));
                DepthType const depth_sq = std::pow(col - half_frame_height, 2) + std::pow(row - half_frame_width, 2)
                        + std::pow((z - depthMap.at<DepthType>(row, col)) / ((pixel_scale != 1) ? pixel_scale : 1), 2);
                depthMap.at<DepthType>(row, col) = std::sqrt(depth_sq) * ((pixel_scale != 1) ? pixel_scale : 1);
            }
        }
        return depthMap;
    }

    sensor_msgs::PointCloud2Ptr getCurrentPointCloud() const
    {
        cv::Mat const currentView = getCurrentView();
        cv::Mat const heightView = this->getScaledImageView(heightMap);

        RGBPointCloudElement* pcIterator = reinterpret_cast<RGBPointCloudElement*>(pc->data.data());
        for (auto row = 0; row < getFrameHeight(); row++) {
            for (auto col = 0; col < getFrameWidth(); col++) {
                pcIterator->data.x = static_cast<float>((col - half_frame_width) * pixel_scale);
                pcIterator->data.y = static_cast<float>(-(row - half_frame_height) * pixel_scale);
                pcIterator->data.z = static_cast<float>(z - heightView.at<DepthType>(row, col));
                auto const color = currentView.at<cv::Vec3b>(row, col);
                for (auto i = 0; i < 3; i++) pcIterator->data.rgb[i] = color[i];
                pcIterator++;
            }
        }
        return pc;
    }
};

}

#endif // IMAGE_UTILS_HPP
