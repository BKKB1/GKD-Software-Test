#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>

template <typename T>
class Matrix {
public:
    int rows, cols;
    std::vector<std::vector<T>> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, std::vector<T>(cols, 0));
    }

    Matrix operator+(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        assert(cols == other.rows);
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int k = 0; k < cols; ++k) {
                T val = data[i][k];
                if (val == 0) continue;
                for (int j = 0; j < other.cols; ++j) {
                    result.data[i][j] += val * other.data[k][j];
                }
            }
        }
        return result;
    }

    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i][j] << "\t";
            }
            std::cout << "\n";
        }
    }
};


typedef Matrix<float> Matrixf;
typedef Matrix<double> Matrixd;

template <typename T>
Matrix<T> relu(const Matrix<T>& m) {
    Matrix<T> result(m.rows, m.cols);
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            result.data[i][j] = std::max(static_cast<T>(0), m.data[i][j]);
        }
    }
    return result;
}

template <typename T>
Matrix<T> softmax(const Matrix<T>& m) {
    assert(m.rows == 1 || m.cols == 1);
    Matrix<T> result(m.rows, m.cols);
    T max_val = m.data[0][0];
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            if (m.data[i][j] > max_val) max_val = m.data[i][j];

    T sum_exp = 0;
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            result.data[i][j] = std::exp(m.data[i][j] - max_val);
            sum_exp += result.data[i][j];
        }
    }
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            result.data[i][j] /= sum_exp;
    return result;
}

template <typename T>
class Model {
public:
    Matrix<T> weight1, bias1, weight2, bias2;

    Model(const Matrix<T>& w1, const Matrix<T>& b1, const Matrix<T>& w2, const Matrix<T>& b2)
        : weight1(w1), bias1(b1), weight2(w2), bias2(b2) {}

    Matrix<T> forward(const Matrix<T>& input) const {
        Matrix<T> temp1 = input * weight1;
        Matrix<T> temp2 = temp1 + bias1;
        Matrix<T> temp3 = relu(temp2);
        Matrix<T> temp4 = temp3 * weight2;
        Matrix<T> temp5 = temp4 + bias2;
        return softmax(temp5);
    }
};