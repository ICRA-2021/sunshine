#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sunshine_msgs/LocalSurprise.h>

static ros::Publisher image_publisher;

void publishImage(sunshine_msgs::LocalSurprise::ConstPtr surpriseMsg)
{
    assert(surpriseMsg->surprise_poses.size() == surpriseMsg->surprise.size() * 2); // 2D data only
    sensor_msgs::Image image;
    auto max_x = 0, max_y = 0;
    auto max_perplexity = 0.0;
    for (auto i = 0ul; i < surpriseMsg->surprise_poses.size(); i += 2) {
        assert(surpriseMsg->surprise_poses[i] >= 0 && surpriseMsg->surprise_poses[i + 1] >= 0);
        max_x = std::max(max_x, surpriseMsg->surprise_poses[i] + 1);
        max_y = std::max(max_y, surpriseMsg->surprise_poses[i + 1] + 1);
        max_perplexity = std::max(max_perplexity, surpriseMsg->surprise[i / 2]);
    }

    image.width = static_cast<uint32_t>(max_x);
    image.height = static_cast<uint32_t>(max_y);
    image.encoding = "mono8";
    image.step = image.width;
    image.data.resize(static_cast<uint64_t>(max_x) * static_cast<uint64_t>(max_y));
    assert(image.data.size() >= surpriseMsg->surprise.size());
    uint64_t idx = 0;
    for (auto const& score : surpriseMsg->surprise) {
        image.data[idx++] = static_cast<uint8_t>((score / max_perplexity + 1.0) * 255. / 2.);
    }
    image_publisher.publish(image);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "perplexity_viz");
    ros::NodeHandle nh("~");
    image_publisher = nh.advertise<sensor_msgs::Image>("image", 10);
    ros::Subscriber perplexity_sub = nh.subscribe<sunshine_msgs::LocalSurprise>("/perplexity", 10, &publishImage);
    ros::spin();
}
