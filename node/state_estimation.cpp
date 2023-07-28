#include <string>
#include <sstream>
#include <vector>
#include <ros/ros.h>
#include "ros/param.h"

#include <std_msgs/Int32MultiArray.h>

#include "Eif.h"


class Data_process
{
private:
	ros::Subscriber bboxes_sub;
	ros::Subscriber camera_depth;
public:
};

std::vector<int> detections;

void bboxes_cb(const std_msgs::Int32MultiArray::ConstPtr& msg)
{
	detections = msg->data;
	/*
	for(int i = 0; i < detections.size(); i++)
		std::cout << detections[i] << " ";
	std::cout << std::endl;
	*/
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "formation");
    ros::NodeHandle nh;

    std::string vehicle;
    std::stringstream ss;
    ros::param::get("vehicle", vehicle);
    ss << "/" << vehicle << "/yolov7/yolov7/boundingBox";
    std::string bbox_topic = ss.str();

    ros::Subscriber bboxes_sub = nh.subscribe<std_msgs::Int32MultiArray>(bbox_topic, 10, bboxes_cb);


    ros::spin();
}