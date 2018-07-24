// From http://wiki.ros.org/image_transport/Tutorials/PublishingImages

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");

  ros::NodeHandle nh("~");
  auto const loop_rate = nh.param<double>("loop_rate", 30);
  auto const topic_name = nh.param<std::string>("image_topic", "/camera/image");
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise(topic_name, 1);
  cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  cv::waitKey(30);
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

  if (loop_rate <= 0) {
      throw std::invalid_argument("loop_rate (Hz) cannot be non-positive!");
  }

  ros::Rate rate(loop_rate);
  while (nh.ok()) {
    pub.publish(msg);
    ros::spinOnce();
    rate.sleep();
  }
}
