#include "MathLib.h"

std::vector<double> MathLib::vectors_multiply(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Vectors must be of the same size for element-wise multiplication." << std::endl;
        return {};
    }
    
    std::vector<double> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }
    return result;
}

std::vector<std::vector<std::vector<double>>> MathLib::M_T_mutiply(const Eigen::MatrixXd& M, const std::vector<std::vector<std::vector<double>>>& T){

    size_t n = M.cols();//size of rows.
    size_t m = M.rows();//size of cols.
    size_t p = T[0][0].size();//size of tensors.

    std::vector<std::vector<std::vector<double>>> T_prime(n, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));

    // 計算 T' 的張量
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < p; ++k) {
                for (size_t l = 0; l < m; ++l) {
                    T_prime[i][j][k] += M(i, l) * T[l][j][k];
                }
            }
        }
    }

    return T_prime;
}

std::vector<std::vector<std::vector<double>>> MathLib::T_M_mutiply(const std::vector<std::vector<std::vector<double>>>& T, const Eigen::MatrixXd& M) {

    size_t n = M.cols();//size of rows.
    size_t m = M.rows();//size of cols.
    size_t p = T[0][0].size();//size of tensors.

    std::vector<std::vector<std::vector<double>>> T_prime(n, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));

    // 計算 T' 的張量
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < p; ++k) {
                for (size_t l = 0; l < m; ++l) {
                    T_prime[i][j][k] += T[i][l][k] * M(l, j);
                }
            }
        }
    }

    return T_prime;
}