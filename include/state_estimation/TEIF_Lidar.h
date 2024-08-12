#ifndef TEIF_LIDAR_H
#define TEIF_LIDAR_H

#include"EIF.h"
#include<Eigen/Dense>
class target_EIF_lidar : public EIF
{
    private:
        int target_state_size;
	    int target_measurement_size;   


        MAV_eigen Mav_curr;

        EIF_data T;
        EIF_data self;
        std::vector<EIF_data> targetWRTneighbors;
        Eigen::MatrixXd tQ;
        Eigen::Vector3d lidarMeasurements;
        std::vector<Eigen::Vector3d> pre_lidarMeasurements;
        Eigen::Vector3d u;
        
    public:
        target_EIF_lidar(int state_size);
        ~target_EIF_lidar();
        void setInitialState();
        void computePredPairs(double delta_t);
        void computeCorrPairs();
        // void computeCorrPairs(); 
        void setMeasurement(Eigen::Vector3d LM);
        void setSEIFpredData(EIF_data self);
        void setFusionPairs(Eigen::MatrixXd fusedP, Eigen::VectorXd fusedX, double time);
        void setEstAcc(Eigen::Vector3d acc);
        void setPreMeasurement(Eigen::Vector3d LM);
        bool checkPreMeasurement(Eigen::Vector3d LM);

        std::vector<EIF_data> getEIFData();
        EIF_data getTgtData();
        EIF_data getSelfData();

        bool filter_init=false;

};

#endif
