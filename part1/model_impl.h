#pragma once
#include "model_base.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>

using json = nlohmann::json;
using namespace std;

template <typename T>
class ModelImpl : public ModelBase {
public:
    Matrix<T> weight1, bias1, weight2, bias2;

    ModelImpl(const Matrix<T>& w1, const Matrix<T>& b1, const Matrix<T>& w2, const Matrix<T>& b2)
        : weight1(w1), bias1(b1), weight2(w2), bias2(b2) {}

    
    Matrix<float> forward(const Matrix<float>& input) const override {
     
        Matrix<T> input_T(input.rows, input.cols);
        for(int i=0; i<input.rows; ++i)
            for(int j=0; j<input.cols; ++j)
                input_T.data[i][j] = static_cast<T>(input.data[i][j]);

        Matrix<T> temp1 = input_T * weight1;
        Matrix<T> temp2 = temp1 + bias1;
        Matrix<T> temp3 = relu(temp2);
        Matrix<T> temp4 = temp3 * weight2;
        Matrix<T> temp5 = temp4 + bias2;
        Matrix<T> out_T = softmax(temp5);

        Matrix<float> result(out_T.rows, out_T.cols);
        for(int i=0; i<out_T.rows; ++i)
            for(int j=0; j<out_T.cols; ++j)
                result.data[i][j] = static_cast<float>(out_T.data[i][j]);
        return result;
    }
};

template <typename T>
void readBinFile(const std::string& filePath, Matrix<T>& mat) {
    std::ifstream binFile(filePath, std::ios::binary | std::ios::in);
    if (!binFile.is_open()) {
        throw std::runtime_error("无法打开二进制文件: " + filePath);
    }
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            T val;
            binFile.read(reinterpret_cast<char*>(&val), sizeof(T));
            mat.data[i][j] = val;
        }
    }
}


inline ModelBase* loadModel(const std::string& folderPath) {
    std::string metaPath = folderPath + "/meta.json";
    std::ifstream metaFile(metaPath);
    if (!metaFile.is_open()) throw std::runtime_error("无法打开 meta.json: " + metaPath);
    
    json metaJson;
metaFile >> metaJson;
metaFile.close();

int w1_r = metaJson["fc1.weight"][0];
int w1_c = metaJson["fc1.weight"][1];
int b1_r = metaJson["fc1.bias"][0];
int b1_c = metaJson["fc1.bias"][1];

int w2_r = metaJson["fc2.weight"][0];
int w2_c = metaJson["fc2.weight"][1];
int b2_r = metaJson["fc2.bias"][0];
int b2_c = metaJson["fc2.bias"][1];

std::string typeStr = metaJson["type"];
bool isPlus = (typeStr == "fp64" || folderPath.find("plus") != std::string::npos);

 if (!isPlus) {
    Matrix<float> w1(w1_r, w1_c), b1(b1_r, b1_c), w2(w2_r, w2_c), b2(b2_r, b2_c);
    readBinFile<float>(folderPath + "/fc1.weight", w1);
    readBinFile<float>(folderPath + "/fc1.bias", b1);
    readBinFile<float>(folderPath + "/fc2.weight", w2);
    readBinFile<float>(folderPath + "/fc2.bias", b2);
    return new ModelImpl<float>(w1, b1, w2, b2);
} else {
    Matrix<double> w1(w1_r, w1_c), b1(b1_r, b1_c), w2(w2_r, w2_c), b2(b2_r, b2_c);
    
    readBinFile<double>(folderPath + "/fc1.weight", w1);
    readBinFile<double>(folderPath + "/fc1.bias", b1);
    readBinFile<double>(folderPath + "/fc2.weight", w2);
    readBinFile<double>(folderPath + "/fc2.bias", b2);
    return new ModelImpl<double>(w1, b1, w2, b2);
}
}