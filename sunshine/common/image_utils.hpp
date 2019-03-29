#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include "utils.hpp"
#include <utility>
#include <cv_bridge/cv_bridge.h>
#include <exception>
#include <sensor_msgs/PointCloud2.h>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedStructInspection"

namespace sunshine {
    typedef double_t DepthType;

    cv::Mat scaleImage(cv::Mat const& image, double const &scale, uint32_t const &interpolation = cv::INTER_CUBIC)
    {
        if (scale == 1) return image;
        assert(scale > 0);
        cv::Mat scaled(std::ceil(image.rows * scale), std::ceil(image.cols * scale), image.type());
        cv::resize(image, scaled, scaled.size(), 0, 0, interpolation);
        return scaled;
    }

    class ImageScanner {
    protected:
        cv::Mat const base_image;
        double const scale_factor;
        double const half_frame_width, half_frame_height;
        double x, y;
    private:

        bool isValidPosition() const
        {
            uint32_t const lx = std::lround((x - half_frame_width) / scale_factor), ly = std::lround(
                    (y - half_frame_height) / scale_factor);
            uint32_t const scaled_width = std::lround(getFrameWidth() / scale_factor), scaled_height = std::lround(
                    getFrameHeight() / scale_factor);
            return lx >= 0 && (lx + scaled_width) < base_image.cols && ly >= 0 &&
                   (ly + scaled_height) < base_image.rows;
        }

        cv::Rect getViewRect() const
        {
            if (!isValidPosition()) {
                uint32_t const lx = std::lround((x - half_frame_width) / scale_factor), ly = std::lround(
                        (y - half_frame_height) / scale_factor);
                uint32_t const scaled_width = std::lround(getFrameWidth() / scale_factor), scaled_height = std::lround(
                        getFrameHeight() / scale_factor);
                throw std::domain_error(
                        "Cannot create valid view from position " + std::to_string(x) + "," + std::to_string(y));
            }
            cv::Rect roi;
            roi.x = static_cast<int>(std::lround((x - half_frame_width) / scale_factor));
            roi.y = static_cast<int>(std::lround((y - half_frame_height) / scale_factor));
            roi.width = static_cast<int>(getFrameWidth() / scale_factor);
            roi.height = static_cast<int>(getFrameHeight() / scale_factor);
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

        cv::Mat const getScaledImageView(cv::Mat const &image) const
        {
            return scaleImage(image(getViewRect()), scale_factor);
        }

    public:
        ImageScanner(cv::Mat image, uint32_t frame_width, uint32_t frame_height, double scale = 1.)
                : base_image(std::move(image)),
                  scale_factor(scale),
                  half_frame_width(frame_width / 2.),
                  half_frame_height(frame_height / 2.),
                  x(half_frame_width),
                  y(half_frame_height)
        {
        }

        virtual ~ImageScanner() = default;

        double getScaleFactor() const
        {
            return scale_factor;
        }

        cv::Mat getCurrentView() const
        {
            return getScaledImageView(base_image);
        }

        virtual void moveTo(double const new_x, double const new_y)
        {
            this->x = new_x;
            this->y = new_y;
        }

        void move(double const dx, double const dy)
        {
            this->moveTo(this->x + dx, this->y + dy);
        }

        virtual double getMinX() const
        {
            return half_frame_width;
        }

        virtual double getMinY() const
        {
            return half_frame_height;
        }

        virtual double getMaxX() const
        {
            return base_image.cols * scale_factor - 1;
        }

        virtual double getMaxY() const
        {
            return base_image.rows * scale_factor - 1;
        }

        virtual double getX() const
        {
            return x;
        }

        virtual double getY() const
        {
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
    cv::Mat getFlatHeightMap(uint32_t width, uint32_t height, DepthType z)
    {
        return cv::Mat(height, width, cvType<DepthType>::value, cv::Scalar(z));

    }

    template<typename DepthType = double>
    class ImageScanner3D : public ImageScanner {
    protected:
        cv::Mat const heightMap;
        DepthType const z;
        double const pixel_scale;
        sensor_msgs::PointCloud2Ptr pc;

    public:

        void moveTo(double x, double y) override
        {
            this->x = x / pixel_scale;
            this->y = y / pixel_scale;
        }

        double getMinX() const override
        {
            return half_frame_width * pixel_scale;
        }

        double getMinY() const override
        {
            return half_frame_height * pixel_scale;
        }

        double getMaxX() const override
        {
            return (base_image.cols * scale_factor - 1) * pixel_scale;
        }

        double getMaxY() const override
        {
            return (base_image.rows * scale_factor - 1) * pixel_scale;
        }

        double getX() const override
        {
            return x * pixel_scale;
        }

        double getY() const override
        {
            return y * pixel_scale;
        }

        ImageScanner3D(const cv::Mat &image, uint32_t frame_width_px, uint32_t frame_height_px, cv::Mat heightMap,
                       double scale = 1., double start_z = 1., double pixel_scale = 1)
                : ImageScanner(image, frame_width_px, frame_height_px, scale),
                  heightMap(scaleImage(heightMap, scale)),
                  z(start_z),
                  pixel_scale(pixel_scale),
                  pc(createPointCloud(getFrameWidth(), getFrameHeight(), "rgb"))
        {
            assert(heightMap.type() == sunshine::cvType<DepthType>::value);
            assert(heightMap.size == image.size);

            auto *pcIterator = reinterpret_cast<RGBPointCloudElement *>(pc->data.data());
            for (auto row = 0; row < getFrameHeight(); row++) {
                for (auto col = 0; col < getFrameWidth(); col++) {
                    pcIterator->data.x = static_cast<float>((col - half_frame_width) * pixel_scale);
                    pcIterator->data.y = static_cast<float>((row - half_frame_height) * pixel_scale);
                    pcIterator++;
                }
            }
        }

        cv::Mat getCurrentDepthView() const
        {
            cv::Mat depthMap = this->getScaledImageView(heightMap).clone();
            for (auto row = 0; row < getFrameHeight(); row++) {
                for (auto col = 0; col < getFrameWidth(); col++) {
                    assert(z >= depthMap.at<DepthType>(row, col));
                    DepthType const depth_sq =
                            std::pow(col - half_frame_height, 2) + std::pow(row - half_frame_width, 2) +
                            std::pow((z - depthMap.at<DepthType>(row, col)) / ((pixel_scale != 1) ? pixel_scale : 1),
                                     2);
                    depthMap.at<DepthType>(row, col) = std::sqrt(depth_sq) * ((pixel_scale != 1) ? pixel_scale : 1);
                }
            }
            return depthMap;
        }

        sensor_msgs::PointCloud2Ptr getCurrentPointCloud() const
        {
            cv::Mat const currentView = getCurrentView();
            cv::Mat const heightView = this->getScaledImageView(heightMap);

            auto *pcIterator = reinterpret_cast<RGBPointCloudElement *>(pc->data.data());
            for (auto row = 0; row < getFrameHeight(); row++) {
                for (auto col = 0; col < getFrameWidth(); col++) {
                    assert(z >= heightView.at<DepthType>(row, col));
                    pcIterator->data.z = static_cast<float>(z - heightView.at<DepthType>(row, col));
                    auto const &color = currentView.at<cv::Vec3b>(row, col);
                    for (auto i = 0; i < 3; i++) pcIterator->data.rgb[i] = color[i];
                    pcIterator++;
                }
            }
            return pc;
        }
    };

}

#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif // IMAGE_UTILS_HPP