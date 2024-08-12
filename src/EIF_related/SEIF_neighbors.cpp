#include "SEIF_neighbors.h"

Self_rel_EIF::Self_rel_EIF()
{
	self_measurement_size = 3;
    neighbor_num_curr = 0;
	EIF_measurement_init(self_state_size, self_measurement_size, &self);
    
    //////////////////////// Covariance Tuning ////////////////////////

    R(0, 0) = 4e-4;
    R(1, 1) = 2e-2;
    R(2, 2) = 2e-2;
}
Self_rel_EIF::~Self_rel_EIF(){}

void Self_rel_EIF::setmeasurements(std::vector<Eigen::Vector4d> CMs)
{
    camerameasurements = CMs;
}

void Self_rel_EIF::setNeighborData(std::vector<EIF_data> robots)
{ 
    neighbor_num_curr = 0;
    neighbor_num_curr = robots.size();
    neighbors_pred = robots;
}

void Self_rel_EIF::setEIFpredData(EIF_data pred)
{
    self = pred;
}

EIF_data Self_rel_EIF::computeCorrPair(Eigen::Vector4d CM, EIF_data& neighbor)
{
    double fx = 1029.477219320806;
    double fy = 1029.477219320806;
	double cx = 960.5;
	double cy = 540.5;
    self.z = CM.segment(0, 3);

    self.s.setZero();
    self.y.setZero();

    double X,Y,Z;
    if(checkPreMeasurement(CM))
    {
        Eigen::MatrixXd R_hat;
		Eigen::Matrix3d R_b2c ;
		R_b2c << 0, 1, 0,
				0, 0, 1,
				1, 0, 0;
                
		Eigen::Matrix3d R_w2c = R_b2c*Mav_eigen_self.R_w2b; ///////////////// rotation problem
		Eigen::Vector3d r_qc_c = R_w2c*(neighbor.X_hat.segment(0, 3) - self.X_hat.segment(0, 3)); 

		X = r_qc_c(0)/r_qc_c(2);
		Y = r_qc_c(1)/r_qc_c(2);
		Z = r_qc_c(2);

		self.h(0) = cam.fx()*X + cam.cx();
		self.h(1) = cam.fy()*Y + cam.cy();
		self.h(2) = Z;

        neighbor.z = self.z;
        neighbor.h = self.h;
        ////////////////////////////////////////////////// derivative w.r.t neighbor //////////////////////////////////////////////////
        neighbor.H.setZero(self_measurement_size, self_state_size);

		neighbor.H(0, 0) = (cam.fx()/Z)*(R_w2c(0, 0) - R_w2c(2, 0)*X);
		neighbor.H(0, 1) = (cam.fx()/Z)*(R_w2c(0, 1) - R_w2c(2, 1)*X);
		neighbor.H(0, 2) = (cam.fx()/Z)*(R_w2c(0, 2) - R_w2c(2, 2)*X);
		neighbor.H(1, 0) = (cam.fy()/Z)*(R_w2c(1, 0) - R_w2c(2, 0)*Y);
		neighbor.H(1, 1) = (cam.fy()/Z)*(R_w2c(1, 1) - R_w2c(2, 1)*Y);
		neighbor.H(1, 2) = (cam.fy()/Z)*(R_w2c(1, 2) - R_w2c(2, 2)*Y);
		neighbor.H(2, 0) = R_w2c(2, 0);
		neighbor.H(2, 1) = R_w2c(2, 1);
		neighbor.H(2, 2) = R_w2c(2, 2);
        ////////////////////////////////////////////////// derivative w.r.t self //////////////////////////////////////////////////
        self.H = -neighbor.H;
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        R_hat = R + neighbor.H*neighbor.P_hat*neighbor.H.transpose();

        self.s = self.H.transpose()*R_hat.inverse()*self.H;
        self.y = self.H.transpose()*R_hat.inverse()*(self.z - self.h + self.H*self.X_hat);
    }

    setPreMeasurement(CM);
    return self;
}

void Self_rel_EIF::computeCorrPairs()
{
    selfWRTneighbors.clear();
    for(int i=0; i< neighbor_num_curr; i++)
    {
        for(int j=0; j<camerameasurements.size(); j++)
            if(camerameasurements[j](3) == neighbors_pred[i].ID)
            {
                // std::cout << "\n\n\n"<<camerameasurements[j] <<"\n\n";

                selfWRTneighbors.push_back(computeCorrPair(camerameasurements[j], neighbors_pred[i]));
                break;
            }
    }
}

std::vector<EIF_data> Self_rel_EIF::getEIFData(){ return selfWRTneighbors;}

void Self_rel_EIF::setPreMeasurement(Eigen::Vector4d CM)
{
    if(pre_camerameasurements.size() == 0)
    {
        pre_camerameasurements.push_back(CM);
    }
    else
    {
        bool found = false;
        for(int i=0; i< pre_camerameasurements.size(); i++)
        {
            if(pre_camerameasurements[i](3) == CM(3))
            {
                pre_camerameasurements[i] = CM;
                found = true;
                break;
            }    
        }
        if(!found)
            pre_camerameasurements.push_back(CM);
    }
}

bool Self_rel_EIF::checkPreMeasurement(Eigen::Vector4d CM)
{
    if(pre_camerameasurements.size() > 0)
    {
        for(int i=0; i<pre_camerameasurements.size(); i++)
        {
            if(pre_camerameasurements[i](3) == CM(3))
            {
                if(pre_camerameasurements[i].segment(0, 3) == CM.segment(0, 3))
                    return false;
                else
                    return true;
            }
        }
    }
    return true;
}