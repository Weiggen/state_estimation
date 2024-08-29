#include "EIF.h"
#include "Camera.h"
#include "MathLib.h"

class target_EIF : public EIF
{
private:

    int target_state_size;
	int target_measurement_size;
    

    Camera cam;
    MathLib mathLib;
    double X;
    double Y;
    double Z;

    EIF_data T;
    EIF_data self;
    Eigen::Vector3d boundingBox;
    Eigen::Vector3d CameraModel;
    MAV_eigen Mav_curr;

    Eigen::Vector3d u;

public:
    target_EIF(int state_size);
    ~target_EIF();
    void setInitialState(Eigen::Vector3d Bbox);
    void computePredPairs(double delta_t);
    void computeCorrPairs();
    void setMeasurement(Eigen::Vector3d bBox);
    void setSEIFpredData(EIF_data self);
    void setFusionPairs(Eigen::MatrixXd fusedP, Eigen::VectorXd fusedX, double time);
    void setEstAcc(Eigen::Vector3d acc);
    void setCamera(Camera camera);

    Eigen::MatrixXd getGradientDensityFnc(Eigen::MatrixXd fusedP, 
                                    Eigen::MatrixXd weightedS, Eigen::VectorXd weightedY,
                                    Eigen::VectorXd weightedXi_hat,
                                    double eta_ij);

    EIF_data getTgtData();
    EIF_data getSelfData();

    bool filter_init;
};