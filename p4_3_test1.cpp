// g++ -std=c++17 pnp_demo.cpp `pkg-config --cflags --libs opencv4` -o pnp_demo
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// -------- 1) 相机内参与畸变：直接用你之前的标定结果 --------
Mat K= (Mat_<double>(3,3)<<
    1777.4091, 0, 710.7598,
    0, 1775.4171, 534.7207,
    0, 0, 1);
Mat dist= (Mat_<double>(1,5)<<
    -0.563142, 0.183051, 0.001964, 0.000925, 0.568833);

// -------- 2) 物体(装甲板)坐标系下的 3D 六点（单位 mm） --------
// 坐标系定义：装甲板中心为(0,0,0)，x向右，y向上，z朝相机（平面近似 z=0）
const double SMALL_ARMOR_WIDTH= 135.0;  // 两灯条中心线水平距
const double SMALL_ARMOR_HEIGHT= 55.0;  // 单根灯条有效高度
vector<Point3d> makeObjectPoints()
{
    double W= SMALL_ARMOR_WIDTH;
    double H= SMALL_ARMOR_HEIGHT;
    vector<Point3d> obj;
    // 左灯条：上、中、下
    obj.push_back(Point3d(-W/2, +H/2, 0)); // L_top
    obj.push_back(Point3d(-W/2,   0 , 0)); // L_mid
    obj.push_back(Point3d(-W/2, -H/2, 0)); // L_bot
    // 右灯条：上、中、下
    obj.push_back(Point3d(+W/2, +H/2, 0)); // R_top
    obj.push_back(Point3d(+W/2,   0 , 0)); // R_mid
    obj.push_back(Point3d(+W/2, -H/2, 0)); // R_bot
    return obj;
}

// -------- 3) 示例：从键盘输入或直接替换为检测到的 2D 六点 --------
// 顺序必须与 3D 完全一致：L_top, L_mid, L_bot, R_top, R_mid, R_bot
vector<Point2d> readImagePointsFromStdin()
{
    vector<Point2d> img(6);
    cout<<"请输入六个像素点(u v)，顺序为：L_top L_mid L_bot R_top R_mid R_bot\n";
    for(int i= 0; i<6; i++){
        double u, v;
        cin>>u>>v;
        img[i]= Point2d(u, v);
    }
    return img;
}

// 计算重投影均方根误差（越小越好，单位：像素）
double reprojRMSE(const vector<Point3d>& obj, const vector<Point2d>& img,
                  const Mat& rvec, const Mat& tvec)
{
    vector<Point2d> proj;
    projectPoints(obj, rvec, tvec, K, dist, proj);
    double se= 0.0;
    for(size_t i= 0; i<proj.size(); i++){
        double dx= proj[i].x- img[i].x;
        double dy= proj[i].y- img[i].y;
        se+= dx*dx+ dy*dy;
    }
    return sqrt(se/ proj.size());
}

// 打印 rvec/tvec/距离
void printPose(const Mat& rvec, const Mat& tvec)
{
    double X= tvec.at<double>(0);
    double Y= tvec.at<double>(1);
    double Z= tvec.at<double>(2);
    double dist_cam_to_armor= sqrt(X*X+ Y*Y+ Z*Z); // 与3D单位一致（mm）

    cout<<"rvec = ["<<rvec.at<double>(0)<<", "<<rvec.at<double>(1)<<", "<<rvec.at<double>(2)<<"]\n";
    cout<<"tvec = ["<<X<<", "<<Y<<", "<<Z<<"] mm\n";
    cout<<"distance = "<<dist_cam_to_armor<<" mm\n";
}

// 选择不同PnP算法进行对比
bool runPnP(const vector<Point3d>& obj, const vector<Point2d>& img, int flag, const string& name)
{
    Mat rvec, tvec;
    bool ok= solvePnP(obj, img, K, dist, rvec, tvec, false, flag);
    cout<<"---- "<<name<<" ----\n";
    if(!ok){
        cout<<"solvePnP 失败\n";
        return false;
    }
    printPose(rvec, tvec);
    double rmse= reprojRMSE(obj, img, rvec, tvec);
    cout<<"reproj RMSE = "<<rmse<<" px\n\n";
    return true;
}

int main()
{
    // 3D 点（固定）
    vector<Point3d> obj= makeObjectPoints();

    // 2D 点（示例：从标准输入读；实战中用你检测出的六点替换）
    vector<Point2d> img= readImagePointsFromStdin();

    // 基本健壮性检查：点数与顺序
    if(obj.size()!=6 || img.size()!=6){
        cerr<<"点数必须为6，并保证与3D顺序一致\n";
        return 1;
    }

    // -------- 4) 对比几种PnP算法 --------
    runPnP(obj, img, SOLVEPNP_ITERATIVE, "SOLVEPNP_ITERATIVE"); // 迭代法，稳、精度高
    runPnP(obj, img, SOLVEPNP_EPNP, "SOLVEPNP_EPNP");           // EPnP，速度快
    runPnP(obj, img, SOLVEPNP_AP3P, "SOLVEPNP_AP3P");           // AP3P，少点数方案
    // 若有外点/偶发错误点，建议改用 solvePnPRansac 做鲁棒估计

    return 0;
}
