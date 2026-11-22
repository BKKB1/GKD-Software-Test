#include "matrix.h"
#include <iostream> 
#include <cmath>
#include <cassert>

bool isClose(float a,float b,float epsilon = 1e-5){
    return std::abs(a-b)<epsilon;
}

void testMatrix0ps(){
    std::cout<<"---start---"<< std::endl;
    
    Matrix P(2,3);
    P.data={{1.8f,3.4f,1.1f},{2.1f,1.8f,3.1f}};
    Matrix Q(2,3);
    Q.data ={{0.1f,0.7f,2.5f},{3.1f,2.9f,1.5f}};

    Matrix R_add=P+Q;
    assert(isClose(R_add.data[0][0],1.9f));
    assert(isClose(R_add.data[1][2],4.6f));
    std::cout<<"+ is ok"<< std::endl;

    Matrix P_mul(2,3);
    P_mul.data={{1.0f,2.0f,1.0f},{0.0f,1.0f,2.0f}};
    Matrix Q_mul(3,3);
    Q_mul.data ={{1.0f,0.0f,2.0f},{0.0f,1.0f,1.0f},{0.0f,1.0f,2.0f}};

    Matrix R_mul=P_mul*Q_mul;
    assert(isClose(R_mul.data[0][1],3.0f));
    assert(isClose(R_mul.data[1][0],0.0f));
    std::cout<<"* is ok"<< std::endl;

    Matrix z(1,3);
    z.data ={{1.0f,2.0f,3.0f}};
    Matrix s_out = softmax(z);
    float sum = s_out.data[0][0]+ s_out.data[0][1]+s_out.data[0][2];
    assert(isClose(sum, 1.0f));
    std::cout<<"Softmax  is ok\n"<<std::endl;

}

int main(){
    testMatrix0ps();
    std::cout<<"---start Model ---"<< std::endl;
    Matrix w1(784,500);
    Matrix b1(1,500);
    Matrix w2(500,10);
    Matrix b2(1,10);

    for(int i=0;i< 784;++i){
        for(int j=0;j<500;++j){
            w1.data[i][j]=0.01f;
        }
    }

    Model myModel(w1,b1,w2,b2);
    Matrix input(1,784);
    for(int j=0;j<784;j++)input.data[0][j]=1.0f;

    Matrix output =myModel.forward(input);
    std::cout<<output.rows<<"x"<<output.cols<<std::endl;
    std::cout<<"result"<<std::endl;
    output.print();
    std::cout<<"\n part1 ok"<<std::endl;
    return 0;
}