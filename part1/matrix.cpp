#include "matrix.h"
#include <cmath>
#include <cassert>
#include <algorithm>

Matrix::Matrix() : rows(0),cols(0){}
Matrix::Matrix(int r, int c) : rows(r), cols(c){
    data.resize(rows,std::vector<float>(cols,0.0f));
}
Matrix Matrix::operator+(const Matrix& other)const{
    assert(rows == other.rows && cols == other.cols);
    Matrix result(rows,cols);
    for(int i=0; i < rows;++i){
        for(int j=0; j <cols;++j){
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}
Matrix Matrix::operator*(const Matrix& other) const {
    assert(cols == other.rows);
    Matrix result(rows, other.cols);
    
    for(int i=0; i <rows;++i){
        for(int k=0; k< cols;++k){
            float val = data[i][k];
            if(val == 0.0f)continue;
            for(int j=0;j< other.cols;++j){
                result.data[i][j] += val * other.data[k][j];
            }
        }
    }
    return result;
}
void Matrix::print()const{
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            std::cout << data[i][j] <<"\t";
        }
        std::cout << "\n";
    }
}
Matrix relu(const Matrix& m){
    Matrix result(m.rows,m.cols);
    for(int i=0;i<m.rows;++i){
        for(int j=0; j<m.cols;++j){
            result.data[i][j] = std::max(0.0f,m.data[i][j]);
        }
    }
    return result;
}

Matrix softmax(const Matrix& m){
    assert(m.rows==1 || m.cols == 1);
    Matrix result(m.rows,m.cols);

    float  max_val = -1e9;
    for(int i=0;i<m.rows;++i)
       for(int j=0;j<m.cols;++j)
          if(m.data[i][j]> max_val)max_val = m.data[i][j];

    float  sum_exp =0.0f;
    for(int i=0;i<m.rows;++i){
       for(int j=0;j<m.cols;++j){
        result.data[i][j]=std::exp(m.data[i][j]- max_val);
        sum_exp += result.data[i][j];
       }
    }
    for(int i=0;i<m.rows;++i){
        for(int j=0;j<m.cols;++j){
            result.data[i][j] /= sum_exp;
        }
    }
    return result;
}

Model::Model(const Matrix& w1, const Matrix& b1, const Matrix& w2,const Matrix& b2)
    : weight1(w1), bias1(b1), weight2(w2),bias2(b2){}
Matrix Model::forward(const Matrix& input){
    Matrix temp1 = input * weight1;
    Matrix temp2 = temp1 + bias1;
    Matrix temp3 = relu(temp2);
    Matrix temp4 = temp3* weight2;
    Matrix temp5 = temp4 + bias2;
    return softmax(temp5);
}