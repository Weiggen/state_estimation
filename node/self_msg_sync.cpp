#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

#include <ros/ros.h>
#include "ros/param.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <yolov8_ros_msgs/BoundingBoxes.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float64MultiArray.h>
#include <state_estimation/Int32MultiArrayStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Imu.h>

using namespace std;
using namespace message_filters;

class Image_process
{
private:
  ros::Publisher sync_yolo_pub;
  ros::Publisher sync_depth_pub;
  ros::Publisher sync_bbox_pub;

  sensor_msgs::Image sync_img_yolo;
  sensor_msgs::Image sync_img_depth;
  std_msgs::Float64MultiArray sync_bbox_msgs;

  string vehicle;
  string yolo_input_topic;
  string yolo_output_topic;
  string depth_input_topic;
  string depth_output_topic;
  string bbox_input_topic;
  string bbox_output_topic;
  

  int bbox_col;
  bool start;

  void sync_cb(const sensor_msgs::ImageConstPtr& ori_yolo,
               const sensor_msgs::ImageConstPtr& ori_depth,
               const yolov8_ros_msgs::BoundingBoxes::ConstPtr& ori_bbox);

public:
  Image_process(ros::NodeHandle &nh, string group_ns, int ID);
  ~Image_process();

  double getDepth(int u, int v);
  void set_topic(string group_ns, int ID);
  void set_bbox_col(int col);
  void reArrangeBbox(yolov8_ros_msgs::BoundingBoxes bbox_msgs);
};

Image_process::Image_process(ros::NodeHandle &nh, string group_ns, int ID)
{
  start = false;
  vehicle = group_ns;
  set_topic(group_ns, ID);

  sync_yolo_pub = nh.advertise<sensor_msgs::Image>(yolo_output_topic, 1);
  sync_depth_pub = nh.advertise<sensor_msgs::Image>(depth_output_topic, 1);
  sync_bbox_pub = nh.advertise<std_msgs::Float64MultiArray>(bbox_output_topic, 1);

  message_filters::Subscriber<sensor_msgs::Image> img_yolo_sub(nh, yolo_input_topic, 1);
  message_filters::Subscriber<sensor_msgs::Image> img_depth_sub(nh, depth_input_topic, 1);
  message_filters::Subscriber<yolov8_ros_msgs::BoundingBoxes> bbox_msg_sub(nh, bbox_input_topic, 1);


  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image,
                                                          yolov8_ros_msgs::BoundingBoxes> MySyncPolicy;

  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), 
                                                   img_yolo_sub,
                                                   img_depth_sub,
                                                   bbox_msg_sub);
  // sync.reset(new sync(MySyncPolicy(100), 
  //                                                  img_yolo_sub,
  //                                                  img_depth_sub,
  //                                                  bbox_msg_sub));                                                   
  sync.registerCallback(boost::bind(&Image_process::sync_cb, this, _1, _2, _3));

  bbox_col = 5;

  ros::spin();
}

Image_process::~Image_process(){}

void Image_process::sync_cb(const sensor_msgs::ImageConstPtr& ori_yolo, 
                            const sensor_msgs::ImageConstPtr& ori_depth,
                            const yolov8_ros_msgs::BoundingBoxes::ConstPtr& ori_bbox)
{
  if(!start)
  {
    start = true;
    cout << "[" << vehicle << " Message_synchronizer]: Start synchronizing messages. \n";
  }

  sync_img_yolo = *ori_yolo;
  sync_img_depth = *ori_depth;
  reArrangeBbox(*ori_bbox);  

  sync_yolo_pub.publish(sync_img_yolo);
  sync_depth_pub.publish(sync_img_depth);
  sync_bbox_pub.publish(sync_bbox_msgs);
}

void Image_process::reArrangeBbox(yolov8_ros_msgs::BoundingBoxes bbox_msgs)
{
  double u, v;
  vector<int> bbox_data;
  if ( !bbox_msgs.bounding_boxes.empty())
  {
    bbox_data.push_back(bbox_msgs.bounding_boxes[0].xmin);
    bbox_data.push_back(bbox_msgs.bounding_boxes[0].ymin);
    bbox_data.push_back(bbox_msgs.bounding_boxes[0].xmax);
    bbox_data.push_back(bbox_msgs.bounding_boxes[0].ymax);
  }
  vector<double> reArrangeBbox_data;

  float depth = 0;
  if( !bbox_data.empty())
  {
    for(int i = 0; i <bbox_data.size(); i+=bbox_col)
    {
      u = (double)(bbox_data[i] + bbox_data[i+2])/2;
      v = (double)(bbox_data[i+1] + bbox_data[i+3])/2;
      reArrangeBbox_data.push_back(u);
      reArrangeBbox_data.push_back(v);
      depth = getDepth((int)u, (int)v);
      if(isnan(depth))
        reArrangeBbox_data.clear();  
      else
        reArrangeBbox_data.push_back(depth);
    }
    sync_bbox_msgs.data = reArrangeBbox_data;
  }
}

double Image_process::getDepth(int u, int v)
{
  float depth = 0;
  int n = 0;
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(sync_img_depth, sensor_msgs::image_encodings::TYPE_32FC1);
  if (u >= 0 && u < cv_ptr->image.cols && v >= 0 && v < cv_ptr->image.rows) {
    depth = cv_ptr->image.at<float>(v, u); // Access depth value at (u, v)
  } 
  
  return depth;
}

void Image_process::set_topic(string group_ns, int ID)
{

  string prefix = string("/")+ group_ns + to_string(ID);

  yolo_input_topic = prefix + string("/yolov8/detection_image");
  depth_input_topic = prefix + string("/camera/depth/image_raw");
  bbox_input_topic = prefix + string("/yolov8/BoundingBox");

  cout << "[" << group_ns << " Message_synchronizer]: Input topic was set:\n"
                          << yolo_input_topic << endl 
                          << depth_input_topic << endl
                          << bbox_input_topic << endl
                          << "==============================================\n";

  yolo_output_topic = prefix + string("/synchronizer/yolov8/visualization");
  depth_output_topic = prefix + string("/synchronizer/camera/depth/image_raw");
  bbox_output_topic = prefix + string("/synchronizer/yolov8/boundingBox");

  cout << "[" << group_ns << " Message_synchronizer]: Output topic was set:\n"
                          << yolo_output_topic << endl 
                          << depth_output_topic << endl
                          << bbox_output_topic << endl
                          << "===================================================================================================\n\n";
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "msg_synchronizer");
  ros::NodeHandle nh;

  string vehicle;
  int ID = 0;
  ros::param::get("vehicle", vehicle);
  ros::param::get("ID", ID);
  // std::cout << "work out \n\n";
  Image_process process(nh, vehicle, ID);
  
  return 0;
}
