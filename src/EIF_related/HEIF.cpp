#include "HEIF.h"

HEIF::HEIF(int x_size=9)
{
	state_size = x_size;
	initialize();
}

HEIF::~HEIF(){}

void HEIF::initialize()
{
	weightedOmega_hat.setZero(state_size, state_size);
	weightedXi_hat.setZero(state_size);
	weightedS.setZero(state_size, state_size);
	weightedY.setZero(state_size);
}

void HEIF::CI_combination()
{
	fusedP = (weightedOmega_hat + weightedS).inverse(); // posterior p ( weightedS = p_breve.inverse() )
	fusedX = fusedP*(weightedXi_hat + weightedY);		// posterior x
}

Eigen::MatrixXd HEIF::getFusedCov(){return fusedP;}
Eigen::VectorXd HEIF::getFusedState(){return fusedX;}

Eigen::MatrixXd HEIF::getWeightedS(){return weightedS;}
Eigen::VectorXd HEIF::getWeightedY(){return weightedY;}
Eigen::VectorXd HEIF::getWeightedXi_hat(){return weightedXi_hat;}