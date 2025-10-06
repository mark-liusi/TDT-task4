#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
int main()
{
    VectorXd x;   // 状态向量
    MatrixXd F;   // 状态转移矩阵
    MatrixXd P;   // 状态协方差矩阵
    MatrixXd Q;   // 过程噪声
    MatrixXd H;   // 观测矩阵
    MatrixXd R;   // 观测噪声
    MatrixXd K;   // 卡尔曼增益

    x = VectorXd(4); // [x, y, vx, vy]
    F = MatrixXd::Identity(4, 4); // 状态转移矩阵
    P = MatrixXd::Identity(4, 4) * 1000; // 初始协方差大一点
    Q = MatrixXd::Identity(4, 4) * 0.01; // 过程噪声
    H = MatrixXd::Zero(2, 4); // 观测矩阵
    H(0,0) = 1; H(1,1) = 1;   // 只观测位置
    R = MatrixXd::Identity(2, 2) * 5; // 观测噪声

    // Prediction
    x = F * x;               // 状态预测
    P = F * P * F.transpose() + Q; // 协方差预测

    // Update
    VectorXd z(2); // 观测值 [x_meas, y_meas]
    VectorXd y = z - H * x;  // 残差
    MatrixXd S = H * P * H.transpose() + R;
    K = P * H.transpose() * S.inverse(); // 卡尔曼增益
    x = x + K * y;          // 状态更新
    P = (MatrixXd::Identity(4,4) - K * H) * P; // 协方差更新


}