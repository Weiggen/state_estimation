#ifndef GT_MEA
#define GT_MEA
#pragma once

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <gazebo_msgs/ModelStates.h>
#include <Eigen/Dense>
#include <random>

#include "Mav.h"

class GT_measurement
{
private:
    ros::NodeHandle nh;
    ros::Subscriber bboxes_sub;
    ros::Subscriber groundTruth_sub;
    
    int self_index;
    int formation_num;
    int mavNum;
    int ID;
    int rosRate;
    /*=================================================================================================================================
        groundtruth
    =================================================================================================================================*/
    MAV* GTs;
	std::vector<MAV_eigen> GTs_eigen;

    int GTs_rate;
    int GTs_count;
    /*=================================================================================================================================
        Lidar, position
    ===============================================================================================================================*/
    std::vector<Eigen::Vector4d> lidarMeasurements;
    Eigen::Vector3d lidar4target;
	Eigen::Vector3d positionMeasurement;

    int lidar_rate;
    int position_rate;

    std::vector<Eigen::Vector4d> lidarMeasure(std::vector<MAV_eigen> GTs_eigen, std::default_random_engine generator);
    Eigen::Vector3d lidarmeasure4target(std::vector<MAV_eigen> formation_eigen_GT, MAV_eigen target_eigen, std::default_random_engine generator);
    Eigen::Vector3d positionMeasure(MAV_eigen GT_eigen, std::default_random_engine generator);
  /*=================================================================================================================================
        Camera model
    =================================================================================================================================*/
    Camera cam;

    std::vector<Eigen::Vector4d> CameraModel;
    Eigen::Vector3d CameraModel4target;
    std::vector<Eigen::Vector4d> Camera4Neighbor(std::vector<MAV_eigen> GTs_eigen, std::default_random_engine generator);
    Eigen::Vector3d CameraMeasure4target(std::vector<MAV_eigen> formation_eigen_GT, MAV_eigen target_eigen, std::default_random_engine generator);
    /*=================================================================================================================================
        Camera boundingBox
    =================================================================================================================================*/
	std::vector<double> bboxes_raw;
	Eigen::Vector3d bbox_eigen;
	Eigen::Vector3d bbox_eigen_past;

    int bbox_count;
    int no_bbox_count;
    int checkCount;
    bool gotBbox;

public:
    GT_measurement(ros::NodeHandle& nh_, int ID, int mavnum);
    ~GT_measurement();
    void setRosRate(int rate);
    /*=================================================================================================================================
        groundtruth
    =================================================================================================================================*/
    void groundTruth_cb(const gazebo_msgs::ModelStates::ConstPtr& msg);
    std::vector<MAV_eigen> getGTs_eigen();
    geometry_msgs::Quaternion getGTorientation(int ID);

    /*=================================================================================================================================
        Lidar, position
    =================================================================================================================================*/
    std::vector<Eigen::Vector4d> getLidarMeasurements();
    Eigen::Vector3d getlidar4target();
    Eigen::Vector3d getPositionMeasurement();

 /*=================================================================================================================================
        Camera model
    =================================================================================================================================*/
    void setCamera(Camera camera);

    std::vector<Eigen::Vector4d>getCameraNeighbor();
    Eigen::Vector3d getCamera4target();
    /*=================================================================================================================================
        Camera boundingBox
    =================================================================================================================================*/
    void bboxes_cb(const std_msgs::Float64MultiArray::ConstPtr& msg);
    bool ifCameraMeasure();
    void bbox_check();
    Eigen::Vector3d getBboxEigen();
};





#endif