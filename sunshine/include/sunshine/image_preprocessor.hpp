#ifndef SUNSHINE_PROJECT_IMAGE_PREPROCESSOR_HPP
#define SUNSHINE_PROJECT_IMAGE_PREPROCESSOR_HPP

#include <cv_bridge/cv_bridge.h>
#include "image_transport/image_transport.h"
#include "ros/ros.h"
#include "sensor_msgs/image_encodings.h"
#include "common/image_processing.hpp"

namespace sunshine {
class ImagePreprocessor {
    bool use_clahe, apply_devignette, correct_colors, show_debug;

    image_transport::ImageTransport it;
    image_transport::Subscriber imageSub;
    image_transport::Publisher imagePub;

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

  public:
    explicit ImagePreprocessor(ros::NodeHandle* nh);
};
} // namespace sunshine

#endif // SUNSHINE_PROJECT_IMAGE_PREPROCESSOR_HPP
