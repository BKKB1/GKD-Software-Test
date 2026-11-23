#include "../part1/matrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

Model loadModelFromFiles(const string& folderPath) {
    string metaPath = folderPath + "/meta.json";
    ifstream metaFile(metaPath);
    if (!metaFile.is_open()) {
        throw runtime_error("无法打开 meta.json 文件，请检查路径：" + metaPath);
    }

    json metaJson;
    metaFile >> metaJson;
    metaFile.close();
    cout << "JSON 文件路径：" << metaPath << endl;
cout << "JSON 实际读取内容：" << endl;
cout << metaJson.dump(4) << endl;

// 检查 JSON 是否包含需要的键
vector<string> keys = {
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias"
};

for (const auto& key : keys) {
    if (!metaJson.contains(key)) {
        throw runtime_error("JSON 缺少字段：" + key);
    }

    if (!metaJson[key].is_array() || metaJson[key].size() != 2) {
        throw runtime_error("JSON 字段格式错误：" + key);
    }

    if (!metaJson[key][0].is_number_integer() ||
        !metaJson[key][1].is_number_integer()) {
        throw runtime_error("JSON 维度不是整数：" + key);
    }
}

int w1_rows = metaJson["fc1.weight"][0].get<int>();
int w1_cols = metaJson["fc1.weight"][1].get<int>();

int b1_rows = metaJson["fc1.bias"][0].get<int>();
int b1_cols = metaJson["fc1.bias"][1].get<int>();

int w2_rows = metaJson["fc2.weight"][0].get<int>();
int w2_cols = metaJson["fc2.weight"][1].get<int>();

int b2_rows = metaJson["fc2.bias"][0].get<int>();
int b2_cols = metaJson["fc2.bias"][1].get<int>();

    Matrix w1(w1_rows, w1_cols);
    Matrix b1(b1_rows, b1_cols);
    Matrix w2(w2_rows, w2_cols);
    Matrix b2(b2_rows, b2_cols);

    auto readBinFile = [](const string& filePath, Matrix& mat) {
        ifstream binFile(filePath, ios::binary | ios::in);
        if (!binFile.is_open()) {
            throw runtime_error("无法打开二进制文件: " + filePath);
        }
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                float val;
                binFile.read(reinterpret_cast<char*>(&val), sizeof(float));
                mat.data[i][j] = val;
            }
        }
        binFile.close();
    };

  readBinFile(folderPath + "/fc1.weight", w1);
readBinFile(folderPath + "/fc1.bias", b1);
readBinFile(folderPath + "/fc2.weight", w2);
readBinFile(folderPath + "/fc2.bias", b2);
    return Model(w1, b1, w2, b2);
}

int main() {
    try {
        cout << "--- 开始加载模型 ---" << endl;
        string modelDir = "../mnist-fc"; 
        
        Model myModel = loadModelFromFiles(modelDir);
        cout << "模型加载成功！" << endl;

        cout << "weight1 尺寸: " << myModel.weight1.rows << " x " << myModel.weight1.cols << endl;
        cout << "weight1[0][0] 的值: " << myModel.weight1.data[0][0] << endl;

        Matrix input(1, 784);
        for(int j=0; j<784; j++) input.data[0][j] = 1.0f; 
        
        cout << "--- 开始前向传播 ---" << endl;
        Matrix output = myModel.forward(input);
        cout << "输出结果大小: " << output.rows << " x " << output.cols << endl;
        output.print();

    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
        return 1;
    }
    return 0;
}