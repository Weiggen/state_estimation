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

    size_t n = T.size();//size of rows.
    size_t m = T[0].size();//size of cols.
    size_t p = T[0][0].size();//size of tensors.

    std::vector<std::vector<std::vector<double>>> T_prime(n, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < p; ++k) {
                for (size_t l = 0; l < m; ++l) {
                    T_prime[i][j][k] = M(i, l) * T[l][j][k];
                }
            }
        }
    }
    return T_prime;
}

std::vector<std::vector<std::vector<double>>> MathLib::T_M_mutiply(const std::vector<std::vector<std::vector<double>>>& T, const Eigen::MatrixXd& M) {

    size_t n = T.size();//size of rows.
    size_t m = T[0].size();//size of cols.
    size_t p = T[0][0].size();//size of tensors.

    std::vector<std::vector<std::vector<double>>> T_prime(n, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < p; ++k) {
                for (size_t l = 0; l < m; ++l) {
                    T_prime[i][j][k] = T[i][l][k] * M(l, j);
                }
            }
        }
    }
    return T_prime;
}

std::vector<std::vector<std::vector<double>>> MathLib::T_transpose(const std::vector<std::vector<std::vector<double>>>& T){

    size_t n = T.size();//size of rows.
    size_t m = T[0].size();//size of cols.
    size_t p = T[0][0].size();//size of tensors.

    std::vector<std::vector<std::vector<double>>> T_prime(n, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < p; ++k) {
                T_prime[i][j][k] = T[j][i][k];
            }
        }
    }
    return T_prime;
}

std::vector<std::vector<std::vector<double>>> MathLib::cT(const double& c, const std::vector<std::vector<std::vector<double>>>& T){

    size_t n = T.size();//size of rows.
    size_t m = T[0].size();//size of cols.
    size_t p = T[0][0].size();//size of tensors.

    std::vector<std::vector<std::vector<double>>> T_prime(n, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < p; ++k) {
                T_prime[i][j][k] = c*T[i][j][k];
            }
        }
    }
    return T_prime;
}

std::vector<std::vector<std::vector<double>>> MathLib::T_addition(const std::vector<std::vector<std::vector<double>>>& T1, const std::vector<std::vector<std::vector<double>>>& T2){

    size_t n = T1.size();//size of rows.
    size_t m = T1[0].size();//size of cols.
    size_t p = T1[0][0].size();//size of tensors.

    std::vector<std::vector<std::vector<double>>> T_prime(n, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < p; ++k) {
                T_prime[i][j][k] = T1[i][j][k] + T2[i][j][k];
            }
        }
    }
    return T_prime;
}

Eigen::MatrixXd MathLib::T_V_mutiply(const std::vector<std::vector<std::vector<double>>>& T, const Eigen::VectorXd& V){

    size_t n = T.size();//size of rows.
    size_t m = T[0].size();//size of cols.
    size_t p = T[0][0].size();//size of tensors.

    Eigen::MatrixXd M(3, 3);
    M.setZero();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < p; ++k) {
                M(i, k) = V[j] * T[i][j][k];
            }
        }
    }
    return M;
}

std::vector<std::vector<std::vector<double>>> MathLib::T_resize(const std::vector<std::vector<std::vector<double>>>& T, const int& Dim){
    // Dimension reduction: m < T.size().
    size_t m = Dim;
    std::vector<std::vector<std::vector<double>>> T_prime(m, std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0)));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < m; ++k) {
                T_prime[i][j][k] = T[i][j][k];
            }
        }
    }
    return T_prime;    
}

Eigen::MatrixXd MathLib::M_resize(const Eigen::MatrixXd& M, const int& Dim){
    size_t m = Dim;
    Eigen::MatrixXd M_prime(m, m);
    M_prime.setZero();
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            M_prime(i, j) = M(i, j);
        }
    }
    return M_prime;
}

Eigen::MatrixXd MathLib::TensorContraction(const std::vector<Eigen::MatrixXd>& multi_M, const std::vector<std::vector<std::vector<double>>>& T){
    size_t n = multi_M.size();
    Eigen::MatrixXd M_prime(2, n);
    M_prime.setZero();
    for (size_t g = 0; g < n; ++g){
        for (size_t k = 0; k < 2; ++k){
            for (size_t i = 0; i < 2; ++i){
                for (size_t j = 0; j < 2; ++j){
                    M_prime(k, g) += multi_M[g](i, j)*T[i][j][k];
                }
            }
        }
    }
    return M_prime;
}