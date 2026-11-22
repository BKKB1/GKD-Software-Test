#pragma once
#include <vector>
#include <iostream>

class Matrix{
public:
    int rows,cols;
    std::vector<std::vector<float>>data;
    
    Matrix();
    Matrix(int r, int c);
    Matrix operator+(const Matrix& other)const;
    Matrix operator*(const Matrix& other)const;
    void print() const;
};
Matrix relu(const Matrix& m);
Matrix softmax(const Matrix& m);

class Model{
public:
    Matrix weight1,bias1,weight2,bias2;
    Model(const Matrix& w1,const Matrix& b1,const Matrix& w2,const Matrix& b2);
    Matrix forward(const Matrix& input);
};