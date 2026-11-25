#include "../part1/model_impl.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

using namespace std;
using namespace std::chrono;

template <typename T>
Matrix<T> matmul_single(const Matrix<T>& A, const Matrix<T>& B) {
    assert(A.cols == B.rows);
    Matrix<T> result(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            T val = A.data[i][k];
            if (val == 0) continue; // 稀疏优化
            for (int j = 0; j < B.cols; ++j) {
                result.data[i][j] += val * B.data[k][j];
            }
        }
    }
    return result;
}

template <typename T>
Matrix<T> matmul_multi(const Matrix<T>& A, const Matrix<T>& B, int num_threads) {
    assert(A.cols == B.rows);
    Matrix<T> result(A.rows, B.cols);
    
    vector<thread> threads;
    int rows_per_thread = A.rows / num_threads;
    int remainder = A.rows % num_threads;

    int start_row = 0;
    for (int t = 0; t < num_threads; ++t) {
        int current_rows = rows_per_thread + (t < remainder ? 1 : 0);
        int end_row = start_row + current_rows;

        threads.emplace_back([&A, &B, &result, start_row, end_row]() {
            for (int i = start_row; i < end_row; ++i) {
                for (int k = 0; k < A.cols; ++k) {
                    T val = A.data[i][k];
                    if (val == 0) continue;
                    for (int j = 0; j < B.cols; ++j) {
                        result.data[i][j] += val * B.data[k][j];
                    }
                }
            }
        });
        
        start_row = end_row;
    }

    for (auto& th : threads) {
        th.join();
    }
    return result;
}

int main() {
    try {
        cout << "--- 加载模型 ---" << endl;
        ModelBase* model = loadModel("../mnist-fc-plus"); 

        int batch_size = 64;
        Matrixf input(batch_size, 784);
        for(int i=0; i<batch_size; i++)
            for(int j=0; j<784; j++) 
                input.data[i][j] = 1.0f; 


        auto start1 = high_resolution_clock::now();
  
        Matrix<float> W(784, 500);
        for(int i=0;i<784;i++) for(int j=0;j<500;j++) W.data[i][j] = 0.01f;
        
        Matrix<float> out1 = matmul_single(input, W);
        
        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(end1 - start1).count();
        cout << "单线程耗时: " << duration1 << " 微秒" << endl;

    
        int cores = thread::hardware_concurrency();
        cout << "检测到 CPU 核心数: " << cores << "，使用多线程计算..." << endl;

        auto start2 = high_resolution_clock::now();
        Matrix<float> out2 = matmul_multi(input, W, cores);
        auto end2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(end2 - start2).count();
        cout << "多线程耗时: " << duration2 << " 微秒" << endl;

        bool match = true;
        for(int i=0; i<out1.rows; i++) {
            for(int j=0; j<out1.cols; j++) {
                if (abs(out1.data[i][j] - out2.data[i][j]) > 1e-5) {
                    match = false; break;
                }
            }
        }
        cout << "结果一致性验证: " << (match ? "通过！" : "失败！") << endl;
        cout << "加速比: " << (double)duration1 / duration2 << " 倍" << endl;

        delete model;
    } catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
    }
    return 0;
}