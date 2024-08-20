#ifndef MATHLIB_H
#define MATHLIB_H

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "MathLib.h"

class MathLib
{
public:
    MathLib();
    ~MathLib();
    std::vector<double> vectors_multiply();
    std::vector<std::vector<std::vector<double>>> M_T_mutiply();
    std::vector<std::vector<std::vector<double>>> T_M_mutiply();
}

#endif