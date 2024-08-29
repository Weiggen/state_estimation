#ifndef MATHLIB_H
#define MATHLIB_H
#pragma once

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "MathLib.h"

class MathLib
{
public:
    std::vector<double> vectors_multiply(const std::vector<double>& vec1, const std::vector<double>& vec2);
    std::vector<std::vector<std::vector<double>>> M_T_mutiply(const Eigen::MatrixXd& M, const std::vector<std::vector<std::vector<double>>>& T);
    std::vector<std::vector<std::vector<double>>> T_M_mutiply(const std::vector<std::vector<std::vector<double>>>& T, const Eigen::MatrixXd& M);
    std::vector<std::vector<std::vector<double>>> T_transpose(const std::vector<std::vector<std::vector<double>>>& T);
    std::vector<std::vector<std::vector<double>>> cT(const double& c,const std::vector<std::vector<std::vector<double>>>& T);
    std::vector<std::vector<std::vector<double>>> T_addition(const std::vector<std::vector<std::vector<double>>>& T1, const std::vector<std::vector<std::vector<double>>>& T2);
    Eigen::MatrixXd T_V_mutiply(const std::vector<std::vector<std::vector<double>>>& T, const Eigen::VectorXd& V);
    std::vector<std::vector<std::vector<double>>> T_resize(const std::vector<std::vector<std::vector<double>>>& T, const int& Dim);
    Eigen::MatrixXd TensorContraction(const std::vector<Eigen::Matrix2d>& multi_M, const std::vector<std::vector<std::vector<double>>>& T);
};

#endif