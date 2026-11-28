#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>

#include "../part1/model_impl.h"

using namespace cv;
using namespace std;

Mat canvas(280, 280, CV_8UC1, Scalar(0)); 
Mat display_img;                          
mutex mtx;                                
bool is_drawing = false;                  
Point last_pt(-1, -1);                    
int current_prediction = -1;              

void on_mouse(int event, int x, int y, int flags, void* userdata) {
    lock_guard<mutex> lock(mtx);
    if (event == EVENT_LBUTTONDOWN) {
        is_drawing = true;
        last_pt = Point(x, y);
    } else if (event == EVENT_MOUSEMOVE && is_drawing) {
        line(canvas, last_pt, Point(x, y), Scalar(255), 15);
        last_pt = Point(x, y);
    } else if (event == EVENT_LBUTTONUP) {
        is_drawing = false;
    }
}

void inference_thread(ModelBase* model) {
    while (true) {
        Mat local_canvas;
        {
            lock_guard<mutex> lock(mtx);
            local_canvas = canvas.clone();
        }
        Mat resized;
        resize(local_canvas, resized, Size(28, 28), 0, 0, INTER_AREA);
        Matrixf input(1, 784);
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                input.data[0][i * 28 + j] = resized.at<uchar>(i, j) / 255.0f;
            }
        }
        Matrixf output = model->forward(input);

        int max_idx = 0;
        float max_prob = output.data[0][0];
        for (int i = 1; i < 10; ++i) {
            if (output.data[0][i] > max_prob) {
                max_prob = output.data[0][i];
                max_idx = i;
            }
        }
        current_prediction = max_idx;

        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

int main() {
    cout << "--- 加载模型 ---" << endl;
    ModelBase* model = loadModel("../mnist-fc");

    thread infer_thread(inference_thread, model);
    infer_thread.detach();

    string win_name = "Handwritten Digit Recognition";
    namedWindow(win_name, WINDOW_AUTOSIZE);
    setMouseCallback(win_name, on_mouse, nullptr);

    cout << "--- 打开画板，请用鼠标画数字 (按 'c' 清空, 按 'ESC' 退出) ---" << endl;

    while (true) {
        Mat combined_img;
        {
            lock_guard<mutex> lock(mtx);
            cvtColor(canvas, display_img, COLOR_GRAY2BGR);
        }

        string text = current_prediction == -1 ? "?" : to_string(current_prediction);
        putText(display_img, "Prediction: " + text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow(win_name, display_img);

        char key = waitKey(30);
        if (key == 27) break; 
        if (key == 'c' || key == 'C') {
            lock_guard<mutex> lock(mtx);
            canvas.setTo(Scalar(0)); 
            current_prediction = -1;
        }
    }

    delete model;
    return 0;
}