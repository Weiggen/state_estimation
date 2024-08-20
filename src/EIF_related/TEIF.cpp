#include "TEIF.h"
#include "HEIF_target.h"
#include "math_function.h"
#include <iostream>
#include <vector>
#include <Eigen/Dense>

target_EIF::target_EIF(int state_size)
{
	target_state_size = state_size;
	target_measurement_size = 3;
	filter_init = false;
	EIF_data_init(target_state_size, target_measurement_size, &T);
	Q.block(0, 0, 3, 3) = 1e-3*Eigen::MatrixXd::Identity(3, 3);
	Q.block(3, 3, 3, 3) = 7e-2*Eigen::MatrixXd::Identity(3, 3);
	// R = 1e-5*Eigen::MatrixXd::Identity(3, 3);
    R(0, 0) = 4e-4;
    R(1, 1) = 4e-4;
    R(2, 2) = 3e-3;
	Mav_curr.v.setZero();
}
target_EIF::~target_EIF(){}

void target_EIF::setInitialState(Eigen::Vector3d Bbox)
{
	Eigen::Matrix3d K;
	K << cam.fx(), 0, cam.cx(),
		0, cam.fy(), cam.cy(),
		0, 0, Bbox(2);
	
	T.X.segment(0, 3) << 0, 0, 5;
	T.X.segment(3, 3) << 0, 0, 0;
	std::cout << "Init:\n" << T.X.segment(0, 3) << std::endl;
	T.P.setIdentity();
	T.P *= 1e-3;
	filter_init = true;
}

void target_EIF::setMeasurement(Eigen::Vector3d bBox){boundingBox = bBox;}

void target_EIF::setSEIFpredData(EIF_data self_data)
{
	self = self_data;
	self.X_hat.segment(0, 3) = self.X_hat.segment(0, 3) + Mav_eigen_self.R_w2b*cam.t_B2C(); ///???????????????????
}

void target_EIF::computePredPairs(double delta_t)
{
	double dt = static_cast<double>(delta_t);
	
	///////////////////////////// X, F ////////////////////////////////

	T.X_hat.segment(0, 3) = T.X.segment(0, 3) + T.X.segment(3, 3)*dt + 1/2*u*dt*dt;
	T.X_hat.segment(3, 3) = T.X.segment(3, 3) + u*dt;

	T.F.setIdentity();
	T.F.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity(3, 3)*dt;

	T.P_hat = T.F*T.P*T.F.transpose() + Q;
}

void target_EIF::computeCorrPairs()
{
	T.z = boundingBox;

	T.s.setZero();
	T.y.setZero();
	self.s.setZero();
	self.y.setZero();
		// std::cout << "[u,d,v] : \n"<< T.z <<"\n\n";

	if(T.z != T.pre_z && T.z(2) >= 2.0 && T.z(2) <= 12.0)
	{
		Eigen::MatrixXd R_hat, R_bar;
		Eigen::Matrix3d R_b2c ;
		R_b2c = cam.R_B2C();

		Eigen::Matrix3d R_w2c = R_b2c*Mav_eigen_self.R_w2b; ///////////////// rotation problem
		Eigen::Vector3d r_qc_c = R_w2c*(T.X_hat.segment(0, 3) - self.X_hat.segment(0, 3)); 

		X = r_qc_c(0)/r_qc_c(2);
		Y = r_qc_c(1)/r_qc_c(2);
		Z = r_qc_c(2);

		T.h(0) = cam.fx()*X + cam.cx();
		T.h(1) = cam.fy()*Y + cam.cy();
		T.h(2) = Z;
		self.z = T.z;
		self.h = T.h;

		T.H(0, 0) = (cam.fx()/Z)*(R_w2c(0, 0) - R_w2c(2, 0)*X);
		T.H(0, 1) = (cam.fx()/Z)*(R_w2c(0, 1) - R_w2c(2, 1)*X);
		T.H(0, 2) = (cam.fx()/Z)*(R_w2c(0, 2) - R_w2c(2, 2)*X);
		T.H(1, 0) = (cam.fy()/Z)*(R_w2c(1, 0) - R_w2c(2, 0)*Y);
		T.H(1, 1) = (cam.fy()/Z)*(R_w2c(1, 1) - R_w2c(2, 1)*Y);
		T.H(1, 2) = (cam.fy()/Z)*(R_w2c(1, 2) - R_w2c(2, 2)*Y);
		T.H(2, 0) = R_w2c(2, 0);
		T.H(2, 1) = R_w2c(2, 1);
		T.H(2, 2) = R_w2c(2, 2);

		self.H = -T.H;
		// T.H    = the partial derivate of the measurement model w.r.t. target pose
		// self.H = the partial derivate of the measurement w.r.t. agent pose

		R_hat = R + self.H*self.P_hat*self.H.transpose();
		R_bar = R + T.H*T.P_hat*T.H.transpose();

		T.s = T.H.transpose()*R_hat.inverse()*T.H;
		T.y = T.H.transpose()*R_hat.inverse()*(T.z - T.h + T.H*T.X_hat);

		self.s = self.H.transpose()*R_bar.inverse()*self.H;
		self.y = self.H.transpose()*R_bar.inverse()*(self.z - self.h + self.H*self.X_hat);
	}
	// T.P = (T.P_hat.inverse() + T.s).inverse();
	// T.X = T.P*(T.P_hat.inverse()*T.X_hat + T.y);
	T.P = T.s.inverse();
	T.X = T.P*T.y;
	T.pre_z = T.z;
}

void target_EIF::computeGradientDensityFnc(Eigen::MatrixXd fusedP, Eigen::MatrixXd weightedS,
										   Eigen::VectorXd weightedXi_hat, Eigen::VectorXd weightedY,
										   double eta_ij)
{
    std::vector<double> gradient_TH_11, gradient_TH_12, gradient_TH_13,
						gradient_TH_21, gradient_TH_22, gradient_TH_23,
						gradient_TH_31, gradient_TH_32, gradient_TH_33;

    Eigen::Matrix3d R_b2c = cam.R_B2C();
    Eigen::Matrix3d R_w2c = R_b2c * Mav_eigen_self.R_w2b;
    Eigen::Vector3d r_qc_c = R_w2c * (T.X_hat.segment(0, 3) - self.X_hat.segment(0, 3)); 

    double X = r_qc_c(0) / r_qc_c(2);
    double Y = r_qc_c(1) / r_qc_c(2);
    double Z = r_qc_c(2);

	// gradient_TH is a 3x3 matrix，using std::vector<double> as its element.
    std::vector<std::vector<std::vector<double>>> gradient_TH(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
    
    gradient_TH_11[0] = (cam.fx() * R_w2c(2, 0) / (Z * Z)) * R_w2c(0, 0) + 
                        ((cam.fx() * R_w2c(0, 0) / (Z * Z)) - (2 * cam.fx() * R_w2c(2, 0) * X / (Z * Z))) * R_w2c(2, 0);
    gradient_TH_11[1] = (cam.fx() * R_w2c(2, 0) / (Z * Z)) * R_w2c(0, 1) + 
                        ((cam.fx() * R_w2c(0, 0) / (Z * Z)) - (2 * cam.fx() * R_w2c(2, 0) * X / (Z * Z))) * R_w2c(2, 1);
    gradient_TH_11[2] = (cam.fx() * R_w2c(2, 0) / (Z * Z)) * R_w2c(0, 2) + 
                        ((cam.fx() * R_w2c(0, 0) / (Z * Z)) - (2 * cam.fx() * R_w2c(2, 0) * X / (Z * Z))) * R_w2c(2, 2);
    
    gradient_TH_12[0] = (cam.fx() * R_w2c(2, 1) / (Z * Z)) * R_w2c(0, 0) + 
                        ((cam.fx() * R_w2c(0, 1) / (Z * Z)) - (2 * cam.fx() * R_w2c(2, 1) * X / (Z * Z))) * R_w2c(2, 0);
    gradient_TH_12[1] = (cam.fx() * R_w2c(2, 1) / (Z * Z)) * R_w2c(0, 1) + 
                        ((cam.fx() * R_w2c(0, 1) / (Z * Z)) - (2 * cam.fx() * R_w2c(2, 1) * X / (Z * Z))) * R_w2c(2, 1);
    gradient_TH_12[2] = (cam.fx() * R_w2c(2, 1) / (Z * Z)) * R_w2c(0, 2) + 
                        ((cam.fx() * R_w2c(0, 1) / (Z * Z)) - (2 * cam.fx() * R_w2c(2, 1) * X / (Z * Z))) * R_w2c(2, 2);

    gradient_TH_13[0] = (cam.fx() * R_w2c(2, 2) / (Z * Z)) * R_w2c(0, 0) + 
                        ((cam.fx() * R_w2c(0, 2) / (Z * Z)) - (2 * cam.fx() * R_w2c(2, 2) * X / (Z * Z))) * R_w2c(2, 0);
    gradient_TH_13[1] = (cam.fx() * R_w2c(2, 2) / (Z * Z)) * R_w2c(0, 1) + 
                        ((cam.fx() * R_w2c(0, 2) / (Z * Z)) - (2 * cam.fx() * R_w2c(2, 2) * X / (Z * Z))) * R_w2c(2, 1);
    gradient_TH_13[2] = (cam.fx() * R_w2c(2, 2) / (Z * Z)) * R_w2c(0, 2) + 
                        ((cam.fx() * R_w2c(0, 2) / (Z * Z)) - (2 * cam.fx() * R_w2c(2, 2) * X / (Z * Z))) * R_w2c(2, 2);

    gradient_TH_21[0] = (cam.fy() * R_w2c(2, 0) / (Z * Z)) * R_w2c(1, 0) + 
                        ((cam.fy() * R_w2c(1, 0) / (Z * Z)) - (2 * cam.fy() * R_w2c(2, 0) * Y / (Z * Z))) * R_w2c(2, 0);
    gradient_TH_21[1] = (cam.fy() * R_w2c(2, 0) / (Z * Z)) * R_w2c(1, 1) + 
                        ((cam.fy() * R_w2c(1, 0) / (Z * Z)) - (2 * cam.fy() * R_w2c(2, 0) * Y / (Z * Z))) * R_w2c(2, 1);
    gradient_TH_21[2] = (cam.fy() * R_w2c(2, 0) / (Z * Z)) * R_w2c(1, 2) + 
                        ((cam.fy() * R_w2c(1, 0) / (Z * Z)) - (2 * cam.fy() * R_w2c(2, 0) * Y / (Z * Z))) * R_w2c(2, 2);

    gradient_TH_22[0] = (cam.fy() * R_w2c(2, 1) / (Z * Z)) * R_w2c(1, 0) + 
                        ((cam.fy() * R_w2c(1, 1) / (Z * Z)) - (2 * cam.fy() * R_w2c(2, 1) * Y / (Z * Z))) * R_w2c(2, 0);
    gradient_TH_22[1] = (cam.fy() * R_w2c(2, 1) / (Z * Z)) * R_w2c(1, 1) + 
                        ((cam.fy() * R_w2c(1, 1) / (Z * Z)) - (2 * cam.fy() * R_w2c(2, 1) * Y / (Z * Z))) * R_w2c(2, 1);
    gradient_TH_22[2] = (cam.fy() * R_w2c(2, 1) / (Z * Z)) * R_w2c(1, 2) + 
                        ((cam.fy() * R_w2c(1, 1) / (Z * Z)) - (2 * cam.fy() * R_w2c(2, 1) * Y / (Z * Z))) * R_w2c(2, 2);
    
    gradient_TH_23[0] = (cam.fy() * R_w2c(2, 2) / (Z * Z)) * R_w2c(1, 0) + 
                        ((cam.fy() * R_w2c(1, 2) / (Z * Z)) - (2 * cam.fy() * R_w2c(2, 2) * Y / (Z * Z))) * R_w2c(2, 0);
    gradient_TH_23[1] = (cam.fy() * R_w2c(2, 2) / (Z * Z)) * R_w2c(1, 1) + 
                        ((cam.fy() * R_w2c(1, 2) / (Z * Z)) - (2 * cam.fy() * R_w2c(2, 2) * Y / (Z * Z))) * R_w2c(2, 1);
    gradient_TH_23[2] = (cam.fy() * R_w2c(2, 2) / (Z * Z)) * R_w2c(1, 2) + 
                        ((cam.fy() * R_w2c(1, 2) / (Z * Z)) - (2 * cam.fy() * R_w2c(2, 2) * Y / (Z * Z))) * R_w2c(2, 2);

    gradient_TH_31 = {0, 0, 0};
    gradient_TH_32 = {0, 0, 0};
    gradient_TH_33 = {0, 0, 0};

    gradient_TH[0][0] = gradient_TH_11;
    gradient_TH[0][1] = gradient_TH_12;
    gradient_TH[0][2] = gradient_TH_13;
    gradient_TH[1][0] = gradient_TH_21;
    gradient_TH[1][1] = gradient_TH_22;
    gradient_TH[1][2] = gradient_TH_23;
    gradient_TH[2][0] = gradient_TH_31;
    gradient_TH[2][1] = gradient_TH_32;
    gradient_TH[2][2] = gradient_TH_33;

	std::vector<std::vector<std::vector<double>>> gradient_H(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));

	for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                gradient_H[i][j][k] = -gradient_TH[i][j][k];
            }
        }
    }

	Eigen::MatrixXd R_tilde, R_bar;

	R_tilde				= R + self.H*self.P_hat*self.H.transpose();//R tilde
	R_bar 				= R + T.H*T.P_hat*T.H.transpose();//R bar

	std::vector<std::vector<std::vector<double>>> gradient_RTilde(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_RTilde_inv(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_TetaTs(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_pBreve(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_p(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_pBreve_inv(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	Eigen::VectorXd x_breve;
	Eigen::MatrixXd gradient_x_breve;
	Eigen::MatrixXd gradient_x_hat;

	gradient_RTilde 	= M_T_multiply(gradient_H, (self.P_hat*self.H.transpose())) + 
						  (self.H*self.P_hat) * gradient_H.transpose();//(17)

	gradient_RTilde_inv = - R_bar.inverse()*gradient_RTilde*R_bar.inverse();//(16)

	gradient_TetaTs 	= eta_ij*(gradient_TH.transpose()*R_hat.inverse()*T.H +
					 	  T.H.transpose()*gradient_RTilde_inv*T.H +
					 	  T.H.transpose()*R_hat.inverse()*gradient_TH);//(12) 
	//eta_ij has not been declared or initialized yet.(in HEIF_target)

	gradient_pBreve 	= - T.s.inverse()*gradient_TetaTs*T.s.inverse();//(11)

	gradient_p 			= fusedP*weightedS.inverse()*gradient_pBreve*weightedS.inverse()*fusedP;//(10)

	gradient_pBreve_inv = - weightedS*gradient_pBreve*weightedS;//(18)

	x_breve				= weightedS.inverse()*weightedY;//(19)

	gradient_x_breve	=

	gradient_x_hat		= gradient_p*(weightedXi_hat + weightedY) +
						  fusedP*gradient_pBreve_inv*p_breve +
						  fusedP*weightedS*gradient_x_breve;//(9)
}

void target_EIF::setFusionPairs(Eigen::MatrixXd fusedP, Eigen::VectorXd fusedX, double time)
{
    T.P = fusedP;
    T.X = fusedX;
}

EIF_data target_EIF::getTgtData(){return T;}
EIF_data target_EIF::getSelfData(){return self;}
void target_EIF::setEstAcc(Eigen::Vector3d acc)
{
	u = acc;
}

void target_EIF::setCamera(Camera camera)
{
	cam = camera;
}