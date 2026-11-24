#include "../part1/model_impl.h"
#include <iostream>

using namespace std;

int main() {
    try {
        Matrixf input(1, 784);
        for(int j=0; j<784; j++) input.data[0][j] = 1.0f; 

        cout << "--- 测试普通模型 (float) ---" << endl;
        ModelBase* model1 = loadModel("../mnist-fc");

        Matrixf out1 = model1->forward(input);
        cout << "输出向量大小: " << out1.rows << " x " << out1.cols << endl;
        out1.print();
        
        delete model1;
        cout << "\n--- 测试 Plus 模型 (double) ---" << endl;
        ModelBase* model2 = loadModel("../mnist-fc-plus");
        
        Matrixf out2 = model2->forward(input);
        cout << "Plus 输出向量大小: " << out2.rows << " x " << out2.cols << endl;
        out2.print();
        
        delete model2;

    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
    }
    return 0;
}