#include"TEIF_Lidar.h"


target_EIF_lidar::target_EIF_lidar(int state_size)
{
	target_state_size = state_size;
	target_measurement_size = 3;
	filter_init = false;
    tQ = 7e-4*Eigen::MatrixXd::Identity(target_state_size, target_state_size);
	EIF_data_init(target_state_size, target_measurement_size, &T);
	tQ.block(0, 0, 3, 3) = 1e-3*Eigen::MatrixXd::Identity(3, 3);
	tQ.block(3, 3, 3, 3) = 8e-2*Eigen::MatrixXd::Identity(3, 3);
	R = 1e-5*Eigen::MatrixXd::Identity(3, 3);

	Mav_curr.v.setZero();
}
target_EIF_lidar::~target_EIF_lidar(){}
void target_EIF_lidar::setInitialState()
{

	T.X.segment(0, 3) << 0, 0, 5;
	T.X.segment(3, 3) << 0, 0, 0;
	std::cout << "Init:\n" << T.X.segment(0, 3) << std::endl;
	T.P.setIdentity();
	T.P *= 1e-3;
    R(0, 0) = 4e-4;
    R(1, 1) = 2e-1;
    R(2, 2) = 2e-1;
	filter_init = true;
}

void target_EIF_lidar::setMeasurement(Eigen::Vector3d LM){lidarMeasurements = LM;}

void target_EIF_lidar::setSEIFpredData(EIF_data self_data)
{
	self = self_data;
	// self.X_hat.segment(0, 3) = self.X_hat.segment(0, 3) + Mav_eigen_self.R_w2b*cam.t_B2C(); ///camera offset
}

void target_EIF_lidar::computePredPairs(double delta_t)
{
	double dt = static_cast<double>(delta_t);
	
	///////////////////////////// X, F ////////////////////////////////

	T.X_hat.segment(0, 3) = T.X.segment(0, 3) + T.X.segment(3, 3)*dt;// + 1/2*u*dt*dt;
	T.X_hat.segment(3, 3) = T.X.segment(3, 3) ;//+ u*dt;

	T.F.setIdentity();

	T.F.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity(3, 3)*dt;
	T.P_hat = T.F*T.P*T.F.transpose() + tQ;
}
void target_EIF_lidar::computeCorrPairs()
{   

    T.z = lidarMeasurements;
    T.s.setZero();
    T.y.setZero();
    self.s.setZero();
    self.y.setZero();

    if (T.z!=T.pre_z )
    {

        Eigen::MatrixXd R_hat,R_bar;
        Eigen::Matrix3d R_W2B = Mav_eigen_self.R_w2b;
        std::cout << self.X_hat.segment(0, 3) <<"\n";
        Eigen::Vector3d r_B_hat = R_W2B*(T.X_hat.segment(0,3) - self.X_hat.segment(0,3));

        double D = sqrt(pow(r_B_hat(0), 2) + pow(r_B_hat(1), 2) + pow(r_B_hat(2), 2));
        T.h(0) = D;
        T.h(1) = std::acos(r_B_hat(2)/D);
        T.h(2) = std::atan2(r_B_hat(1),r_B_hat(0));

        ////////////////////////////////////////////////// derivative w.r.t neighbor //////////////////////////////////////////////////

        T.H.setZero(target_measurement_size,target_state_size);

        T.H(0,0) =  (R_W2B(0, 0)*r_B_hat(0) + R_W2B(1, 0)*r_B_hat(1) + R_W2B(2, 0)*r_B_hat(2)) / D;
        T.H(0, 1) = (R_W2B(0, 1)*r_B_hat(0) + R_W2B(1, 1)*r_B_hat(1) + R_W2B(2, 1)*r_B_hat(2)) / D;
        T.H(0, 2) = (R_W2B(0, 2)*r_B_hat(0) + R_W2B(1, 2)*r_B_hat(1) + R_W2B(2, 2)*r_B_hat(2)) / D;
        
        T.H(1, 0) = (R_W2B(0, 0)*r_B_hat(0)*r_B_hat(2)
                                + R_W2B(1, 0)*r_B_hat(1)*r_B_hat(2)
                                - R_W2B(2, 0)*(r_B_hat(0)*r_B_hat(0) + r_B_hat(1)*r_B_hat(1)))
                                /(D*D * sqrt(r_B_hat(0)*r_B_hat(0) + r_B_hat(1)*r_B_hat(1)));
        T.H(1, 1) = (R_W2B(0, 1)*r_B_hat(0)*r_B_hat(2)
                                + R_W2B(1, 1)*r_B_hat(1)*r_B_hat(2)
                                - R_W2B(2, 1)*(r_B_hat(0)*r_B_hat(0) + r_B_hat(1)*r_B_hat(1)))
                                /(D*D * sqrt(r_B_hat(0)*r_B_hat(0) + r_B_hat(1)*r_B_hat(1)));
        T.H(1, 2) = (R_W2B(0, 2)*r_B_hat(0)*r_B_hat(2)
                                + R_W2B(1, 2)*r_B_hat(1)*r_B_hat(2)
                                - R_W2B(2, 2)*(r_B_hat(0)*r_B_hat(0) + r_B_hat(1)*r_B_hat(1)))
                                /(D*D * sqrt(r_B_hat(0)*r_B_hat(0) + r_B_hat(1)*r_B_hat(1)));

        T.H(2, 0) = (-R_W2B(0, 0)*r_B_hat(1) + R_W2B(1, 0)*r_B_hat(0)) / (r_B_hat(0)*r_B_hat(0) + r_B_hat(1)*r_B_hat(1));
        T.H(2, 1) = (-R_W2B(0, 1)*r_B_hat(1) + R_W2B(1, 1)*r_B_hat(0)) / (r_B_hat(0)*r_B_hat(0) + r_B_hat(1)*r_B_hat(1));
        T.H(2, 2) = (-R_W2B(0, 2)*r_B_hat(1) + R_W2B(1, 2)*r_B_hat(0)) / (r_B_hat(0)*r_B_hat(0) + r_B_hat(1)*r_B_hat(1));

        ////////////////////////////////////////////////// derivative w.r.t self //////////////////////////////////////////////////
        self.H = -T.H; 
		R_hat = R + self.H*self.P_hat*self.H.transpose();
		R_bar = R + T.H*T.P_hat*T.H.transpose();

		T.s = T.H.transpose()*R_hat.inverse()*T.H;
		T.y = T.H.transpose()*R_hat.inverse()*(T.z - T.h + T.H*T.X_hat);

		// self.s = self.H.transpose()*R_bar.inverse()*self.H;
		// self.y = self.H.transpose()*R_bar.inverse()*(self.z - self.h + self.H*self.X_hat);
	}

	T.P = (T.P_hat.inverse() + T.s).inverse();
	T.X = T.P*(T.P_hat.inverse()*T.X_hat + T.y);
	T.pre_z = T.z;
    
}
// void target_EIF_lidar::computeCorrPairs()
// {
//     targetWRTneighbors.clear();
//     // for(int i=0; i< Mav_curr; i++)
//     // {
//         for(int j=0; j<lidarMeasurements.size(); j++)
//             // if(lidarMeasurements[j](3) == T[i].ID)
//             // {
//                 targetWRTneighbors.push_back(computeCorrPair(lidarMeasurements[j]));
//                 // break;
//             // }
//     // }
// }
// std::vector<EIF_data> target_EIF_lidar::getEIFData(){ return targetWRTneighbors;}


void target_EIF_lidar::setFusionPairs(Eigen::MatrixXd fusedP, Eigen::VectorXd fusedX, double time)
{
    T.P = fusedP;
    T.X = fusedX;
}

EIF_data target_EIF_lidar::getTgtData(){return T;}
EIF_data target_EIF_lidar::getSelfData(){return self;}
void target_EIF_lidar::setEstAcc(Eigen::Vector3d acc)
{
	u = acc;
}