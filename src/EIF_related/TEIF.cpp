#include "TEIF.h"
#include "HEIF_target.h"
#include "MathLib.h"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

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
		Eigen::MatrixXd R_tilde, R_bar;
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

		R_tilde = R + self.H*self.P_hat*self.H.transpose();
		R_bar = R + T.H*T.P_hat*T.H.transpose();

		T.s = T.H.transpose()*R_tilde.inverse()*T.H;
		T.y = T.H.transpose()*R_tilde.inverse()*(T.z - T.h + T.H*T.X_hat);

		self.s = self.H.transpose()*R_bar.inverse()*self.H;
		self.y = self.H.transpose()*R_bar.inverse()*(self.z - self.h + self.H*self.X_hat);
	}
	// T.P = (T.P_hat.inverse() + T.s).inverse();
	// T.X = T.P*(T.P_hat.inverse()*T.X_hat + T.y);
	T.P = T.s.inverse();
	T.X = T.P*T.y;
	T.pre_z = T.z;
}

Eigen::MatrixXd target_EIF::getGradientDensityFnc(Eigen::MatrixXd fusedP, Eigen::MatrixXd weightedS, Eigen::VectorXd weightedY, Eigen::VectorXd weightedXi_hat, double eta_ij)
{
    std::vector<double> gradient_TH_11(3), gradient_TH_12(3), gradient_TH_13(3),
						gradient_TH_21(3), gradient_TH_22(3), gradient_TH_23(3),
						gradient_TH_31(3), gradient_TH_32(3), gradient_TH_33(3);

    Eigen::Matrix3d R_b2c;
	R_b2c << 0, 1, 0,
			 0, 0, 1,
			 1, 0, 0;
    Eigen::Matrix3d R_w2c = R_b2c * Mav_eigen_self.R_w2b;
    Eigen::Vector3d r_qc_c = R_w2c * (T.X_hat.segment(0, 3) - self.X_hat.segment(0, 3)); 

    double X = r_qc_c(0) / r_qc_c(2);
    double Y = r_qc_c(1) / r_qc_c(2);
    double Z = r_qc_c(2);

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

    gradient_TH_31 = {0., 0., 0.};
    gradient_TH_32 = {0., 0., 0.};
    gradient_TH_33 = {0., 0., 0.};

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

	Eigen::Matrix3d R_tilde, R_bar;
	Eigen::Matrix3d sP_hat_3D = self.P_hat.block<3, 3>(0, 0);
	Eigen::Matrix3d tP_hat_3D = T.P_hat.block<3, 3>(0, 0);
	Eigen::Matrix3d ts = T.s.block<3, 3>(0, 0);
	Eigen::Matrix3d weightedS_3D = weightedS.block<3, 3>(0, 0);
	Eigen::Vector3d weightedY_3D = weightedY.segment(0, 3);
	Eigen::Matrix3d fusedP_3D = fusedP.block<3, 3>(0, 0);
	Eigen::Vector3d weightedXi_hat_3D = weightedXi_hat.segment(0, 3);

	R_tilde				= R + self.H.block<3, 3>(0, 0)*sP_hat_3D*self.H.block<3, 3>(0, 0).transpose();//R tilde
	R_bar 				= R + T.H.block<3, 3>(0, 0)*tP_hat_3D*T.H.block<3, 3>(0, 0).transpose();//R bar
	// ---------------------------------------------------ok line--------------------------------------------------- //

	std::vector<std::vector<std::vector<double>>> gradient_RTilde(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_RTilde_inv(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_TetaTs(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_pBreve(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_p(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_pBreve_inv(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	std::vector<std::vector<std::vector<double>>> gradient_TH_trans(3, std::vector<std::vector<double>>(3, std::vector<double>(3)));
	Eigen::Matrix3d gradient_x_breve;
	Eigen::Matrix3d gradient_x_hat;
	Eigen::Matrix3d gradient_weightedY;

	// "gradient" means the partial derivate wrt prior pose of the agent(x_bar). //
	gradient_TH_trans 	= mathLib.T_transpose(gradient_TH);

	// std::cout << "gradient_TH_trans:" << std::endl;
	// for (int i = 0; i < 3; ++i) {
	// 	for (int j = 0; j < 3; ++j) {
	// 		for (int k = 0; k < 3; ++k) {
	// 			std::cout << "gradient_TH_trans[" << i << "][" << j << "][" << k << "] = " << gradient_TH_trans[i][j][k] << std::endl;
	// 		}
	// 	}
	// }

	gradient_RTilde 	= mathLib.T_addition(mathLib.T_M_mutiply(gradient_H, (sP_hat_3D*self.H.block<3, 3>(0, 0).transpose())), mathLib.M_T_mutiply((self.H.block<3, 3>(0, 0)*sP_hat_3D), mathLib.T_transpose(gradient_H)));//(17)

	gradient_RTilde_inv = mathLib.T_M_mutiply(mathLib.M_T_mutiply(-R_bar.inverse(), gradient_RTilde), R_bar.inverse());//(16)

	gradient_TetaTs 	= mathLib.cT(eta_ij, (mathLib.T_addition(mathLib.T_addition(mathLib.T_M_mutiply(gradient_TH_trans, R_tilde.inverse()*T.H.block<3, 3>(0, 0)) ,
					 	  mathLib.T_M_mutiply(mathLib.M_T_mutiply(T.H.block<3, 3>(0, 0).transpose(), gradient_RTilde_inv), T.H.block<3, 3>(0, 0))) ,
					 	  mathLib.M_T_mutiply(T.H.block<3, 3>(0, 0).transpose()*R_tilde.inverse(), gradient_TH))));//(12) // eta_ij, from getEta_ij() in HEIF_target.

	gradient_pBreve 	= mathLib.T_M_mutiply(mathLib.M_T_mutiply(- ts.inverse(), gradient_TetaTs), ts.inverse());//(11)

	gradient_p 			= mathLib.T_M_mutiply(mathLib.M_T_mutiply(fusedP_3D*weightedS_3D.inverse(), gradient_pBreve), weightedS_3D.inverse()*fusedP_3D);//(10)

	gradient_pBreve_inv = mathLib.T_M_mutiply(mathLib.M_T_mutiply(- weightedS_3D, gradient_pBreve), weightedS_3D);//(18)

	gradient_weightedY 	= mathLib.T_V_mutiply(mathLib.T_M_mutiply(gradient_TH_trans, R_tilde.inverse()), (T.z-T.h.segment(0, 3)+T.H.block<3, 3>(0, 0)*T.X_hat.segment(0, 3)))+
							mathLib.T_V_mutiply((mathLib.M_T_mutiply(T.H.block<3, 3>(0, 0).transpose(), gradient_RTilde_inv)), (T.z-T.h.segment(0, 3)+T.H.block<3, 3>(0, 0)*T.X_hat.segment(0, 3)))+
							T.H.block<3, 3>(0, 0).transpose()*R_tilde.inverse()*mathLib.T_V_mutiply(gradient_TH, T.X_hat.segment(0, 3));//(19)

	gradient_x_breve	= mathLib.T_V_mutiply(gradient_pBreve, weightedY_3D) + eta_ij*ts.inverse()*gradient_weightedY;//(20)

	gradient_x_hat		= mathLib.T_V_mutiply(gradient_p, (weightedXi_hat_3D + weightedY_3D)) +
						  fusedP_3D*mathLib.T_V_mutiply(gradient_pBreve_inv, (weightedS_3D.inverse()*weightedY_3D)) +
						  fusedP_3D*weightedS_3D*gradient_x_breve;//(9) //weightedXi_hat_3D = q_{T_ij}; //p_breve = weightedY_3D.inverse(); //x_breve = weightedS_3D.inverse()*weightedY_3D;

	// Create meshgrid. --- //
	Eigen::Vector2d grid_size(0.1, 0.1);
	Eigen::Vector2d map_size(24., 24.);
	Eigen::Vector2i size;
	size[0] = static_cast<int>(std::trunc(map_size[0] / grid_size[0]));
	size[1] = static_cast<int>(std::trunc(map_size[1] / grid_size[1]));
	Eigen::MatrixXd x_coords(size[0], size[1]);
	Eigen::MatrixXd y_coords(size[0], size[1]);
	for (int i = 0; i < size[0]; ++i) {
        for (int j = 0; j < size[1]; ++j) {
            x_coords(i, j) = i;
            y_coords(i, j) = j;
        }
    }
	auto q_x = x_coords*grid_size[0];
	auto q_y = y_coords*grid_size[1];
	Eigen::MatrixXd q(2, size[0]*size[1]);
    int k = 0;
    for (int i = 0; i < size[0]; ++i) {
        for (int j = 0; j < size[1]; ++j) {
            q(0, k) = q_x(i, j);
            q(1, k) = q_y(i, j);
            ++k;
        }
    }
	// --- Create meshgrid. //
	
	// pose: 3D -> 2D --- //
	Eigen::Matrix2d weightedS_2D;
	Eigen::Vector2d X_hat_2D;
	Eigen::MatrixXd D_Lphi_x_hat(2, size[0]*size[1]);// (< dimention of vectors >, < numbers of q >)
	std::vector<Eigen::Matrix2d> D_Lphi_p_breve(size[0]*size[1], Eigen::Matrix2d::Zero()); // (< numbers of q >, <dimention of the matrices>)
	weightedS_2D = weightedS.block<2, 2>(0, 0);
	X_hat_2D = T.X_hat.segment(0, 2);
	Eigen::Matrix2d gradient_x_hat_2D;
	gradient_x_hat_2D = gradient_x_hat.block<2, 2>(0, 0);
	// --- 3D -> 2D //

	for (int i = 0; i < q.cols(); ++i){
		D_Lphi_x_hat.col(i) = 0.5 * (weightedS_2D.transpose() * q.col(i) + weightedS_2D * q.col(i) - 2 * weightedS_2D * X_hat_2D);// (8)
    }
	
    for (int i = 0; i < q.cols(); ++i) {
        Eigen::VectorXd dist = q.col(i) - X_hat_2D;
        D_Lphi_p_breve[i] = -0.5 * weightedS_2D + 0.5 * weightedS_2D * dist * dist.transpose() * weightedS_2D;// (21)
    }

	Eigen::MatrixXd multi_vector_result1(2, size[0]*size[1]);
	Eigen::MatrixXd multi_vector_result2(2, size[0]*size[1]);
	multi_vector_result1.setZero();
	multi_vector_result2.setZero();
	multi_vector_result1 = mathLib.TensorContraction(D_Lphi_p_breve, gradient_pBreve);
	multi_vector_result2 = gradient_x_hat_2D.transpose()*D_Lphi_x_hat;

	// std::cout << "multi_vector_result1:\n" << multi_vector_result1 << std::endl;
	// std::cout << "multi_vector_result2:\n" << multi_vector_result2 << std::endl;

	Eigen::MatrixXd gradient2(2, size[0]*size[1]);
	gradient2 = multi_vector_result1 + multi_vector_result2;// (7)

	return gradient2;
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