//
// Created by stewart on 3/9/20.
//

#include "sunshine/image_preprocessor.hpp"


int main(int argc, char** argv) {
    //    std::this_thread::sleep_for(std::chrono::seconds(8));
    // Setup ROS node
    ros::init(argc, argv, "image_preprocessor");
    ros::NodeHandle nh("~");

    sunshine::ImagePreprocessor preprocessor(&nh);
    ros::spin();

    return 0;
}

sunshine::ImagePreprocessor::ImagePreprocessor(ros::NodeHandle* nh)
        : it(*nh)
        , imageSub(it.subscribe("/camera/image_rect_color", 3, &ImagePreprocessor::imageCallback, this))
        , imagePub(it.advertise("/camera/processed_image", 3)) {
    use_clahe        = nh->param<bool>("use_clahe", false);
    correct_colors   = nh->param<bool>("color_correction", false);
    show_debug       = nh->param<bool>("show_debug", false);
    apply_devignette = nh->param<bool>("devignette", false);
}

void sunshine::ImagePreprocessor::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImageConstPtr img_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);

    cv_bridge::CvImage outputImg(img_ptr->header, img_ptr->encoding, cv::Mat());
    cv::Mat& img = outputImg.image;
    if (show_debug) {
        cv::imshow("Original", img);
        cv::waitKey(5);
    }
    if (apply_devignette) {
        img = devignette(img);
        if (show_debug) {
            cv::imshow("Devignetted", img);
            cv::waitKey(5);
        }
    }
    if (correct_colors) {
        img = color_correct(img);
        if (show_debug) {
            cv::imshow("Color Corrected", img);
            cv::waitKey(5);
        }
    }
    if (use_clahe) {
        img = apply_clahe(img);
        if (show_debug) {
            cv::imshow("CLAHE", img);
            cv::waitKey(5);
        }
    }

    imagePub.publish(outputImg.toImageMsg());
}
