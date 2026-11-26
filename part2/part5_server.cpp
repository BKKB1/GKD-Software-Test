#include "../part1/model_impl.h"
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <vector>

using namespace std;

const int PORT = 8080;

int main() {
    cout << "--- 加载模型 ---" << endl;
    ModelBase* model = loadModel("../mnist-fc");
    if (!model) return -1;

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        perror("Socket creation failed");
        return -1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        return -1;
    }

    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        return -1;
    }
    cout << "服务端已启动，正在监听端口 " << PORT << "..." << endl;

    int addrlen = sizeof(address);
    int new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
    if (new_socket < 0) {
        perror("Accept failed");
        return -1;
    }
    cout << "客户端已连接！开始接收数据..." << endl;

    int rows, cols;
    read(new_socket, &rows, sizeof(int));
    read(new_socket, &cols, sizeof(int));
    cout << "收到矩阵尺寸: " << rows << " x " << cols << endl;

    Matrixf input(rows, cols);
    int total_elements = rows * cols;
    int bytes_read = 0;
    int total_bytes = total_elements * sizeof(float);
    vector<float> buffer(total_elements);

    while (bytes_read < total_bytes) {
        int n = read(new_socket, (char*)buffer.data() + bytes_read, total_bytes - bytes_read);
        if (n <= 0) break; 
        bytes_read += n;
    }
    cout << "成功接收 " << bytes_read << " 字节数据。" << endl;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            input.data[i][j] = buffer[i * cols + j];
        }
    }

    cout << "开始进行神经网络推理..." << endl;
    Matrixf output = model->forward(input);

    int out_rows = output.rows;
    int out_cols = output.cols;
    write(new_socket, &out_rows, sizeof(int));
    write(new_socket, &out_cols, sizeof(int));
    
    vector<float> out_buffer(out_rows * out_cols);
    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            out_buffer[i * out_cols + j] = output.data[i][j];
        }
    }
    write(new_socket, out_buffer.data(), out_buffer.size() * sizeof(float));

    cout << "结果已发送。断开连接。" << endl;

    close(new_socket);
    close(server_fd);
    delete model;

    return 0;
}