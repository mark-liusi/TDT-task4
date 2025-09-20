#include <ATen/ATen.h> // Tensor 基本操作
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h> // TorchScript 加载/推理
#include <vector>

using namespace std;
using namespace cv;

class Armor {
public:
    Armor(): id(-1), distance(0) {}
    int id;        // 数字ID
    Mat rvec;      // 旋转向量
    Mat tvec;      // 平移向量
    double distance; // 相机到装甲板中心距离
    double reproj;
};
struct DigitDet {
    Rect box;      // 数字的矩形框（来自 rr）
    string label;  // 数字字符串
};
static vector<DigitDet> digits_this_frame;
static std::unordered_map<int, Armor> g_armor_map;//哈希表容器，使装甲板id和装甲板对象对应起来，按id来保存装甲板的信息
Mat matrix= (Mat_<double>(3,3)<<
    1777.4091, 0, 710.7598,
    0, 1775.4171, 534.7207,
    0, 0, 1);
Mat dist= (Mat_<double>(1,5)<<
    -0.563142, 0.183051, 0.001964, 0.000925, 0.568833);

vector<Point3d> makeObjectPoints()//灯条的3d信息，也就是实际测量信息
{
    double W= 135.0;//宽
    double H= 55.0;//高
    vector<Point3d> obj;
    // 左灯条：上、中、下
    obj.push_back(Point3d(-W/2, +H/2, 0)); 
    obj.push_back(Point3d(-W/2,   0 , 0)); 
    obj.push_back(Point3d(-W/2, -H/2, 0)); 
    // 右灯条：上、中、下
    obj.push_back(Point3d(+W/2, +H/2, 0)); 
    obj.push_back(Point3d(+W/2,   0 , 0)); 
    obj.push_back(Point3d(+W/2, -H/2, 0)); 
    return obj;
}

// using namespace torch;
//  ====== 全局：模型/标签/预处理参数 ======
static torch::jit::script::Module g_module; // 已加载的 TorchScript 模型
static vector<string> g_labels;             // 数字的类别
static int g_in_w = 28, g_in_h = 28; // 模型输入尺寸，训练时为28*28
static bool g_to_gray = true; // 训练时用的是灰度图，勾选为true
static vector<double> g_mean = {
    0.1307}; // 训练时用的参数，为（0.1307， 0.3081）
static vector<double> g_std = {0.3081};
static torch::Device g_device = torch::kCPU; // 我的电脑只有cpu

// 读取标签文件（每行一个标签，索引与训练一致）
static vector<string> load_labels(const string &path) {
  vector<string> labels;
  ifstream fin(path);
  string line;
  while (getline(fin, line)) {
    if (!line.empty())
      labels.push_back(line);
  }
  return labels;
}

// 预处理：把 ROI(Mat) -> Tensor([1,C,H,W])，并做归一化
static torch::Tensor preprocess(Mat roi) {
  Mat img_resized, img_conv;
  resize(
      roi, img_resized,
      Size(g_in_w, g_in_h)); // 将我们裁剪的图像转变为和模型处理参数一致的图像
  if (g_to_gray) {
    cvtColor(img_resized, img_conv,
             COLOR_BGR2GRAY); // BGR->Gray，灰度图，和模型处理一致
  } else {
    cvtColor(img_resized, img_conv, COLOR_BGR2RGB); // BGR->RGB
  }
  img_conv.convertTo(img_conv, CV_32F, 1.0 / 255.0); // [0,1]float

  torch::Tensor t;
  // [H,W] -> [1,H,W]
  t = torch::from_blob(img_conv.data, {img_conv.rows, img_conv.cols},
                       torch::kFloat32).clone();
  t = t.unsqueeze(0);                           //添加通道维度
  t = (t - (float)g_mean[0]) / (float)g_std[0]; // 归一化
  t = t.unsqueeze(0);                           // [1,C,H,W]添加梯次维度
  return t.to(g_device);
}

// 对一个 ROI 做一次推理，返回 (label, conf)
static pair<string, float> infer_one(Mat roi) {
  torch::InferenceMode no_grad;                               // 关闭梯度
  torch::Tensor tin = preprocess(roi);                        // 预处理
  vector<torch::jit::IValue> inputs = {tin};                  // 封装输入
  torch::Tensor logits = g_module.forward(inputs).toTensor(); // 前向
  int pred_idx = logits.argmax(1).item<int>();                // 取最大索引
  torch::Tensor probs = torch::softmax(logits, 1);            // 概率
  float conf = probs[0][pred_idx].item<float>();              // 置信度
  string label = (pred_idx >= 0 && pred_idx < (int)g_labels.size())
                     ? g_labels[pred_idx]
                     : "NA";
  return {label, conf};
}
//通过rvec, tvec来计算重投影误差，比较投影结果和原始输入的 2D 点之间的像素偏差。
double reprojRMSE(const vector<Point3d>& obj, const vector<Point2d>& img,
                  const Mat& rvec, const Mat& tvec)
{
    vector<Point2d> proj;
    projectPoints(obj, rvec, tvec, matrix, dist, proj);
    double se= 0.0;
    for(size_t i= 0; i<proj.size(); i++){
        double dx= proj[i].x- img[i].x;
        double dy= proj[i].y- img[i].y;
        se+= dx*dx+ dy*dy;
    }
    return sqrt(se/ proj.size());
}

void printPose(const Mat& rvec, const Mat& tvec, int armor_id)// 打印revc和tvec和离镜头距离的信息
{
    double X= tvec.at<double>(0);
    double Y= tvec.at<double>(1);
    double Z= tvec.at<double>(2);
    double dist_cam_to_armor= sqrt(X*X+ Y*Y+ Z*Z); // 与3D单位一致（mm）
    Armor &A= g_armor_map[armor_id]; // 不存在会自动创建
    A.id= armor_id;
    A.rvec= rvec.clone();
    A.tvec= tvec.clone();
    A.distance= dist_cam_to_armor;

    cout<<"rvec = ["<<rvec.at<double>(0)<<", "<<rvec.at<double>(1)<<", "<<rvec.at<double>(2)<<"]\n";
    cout<<"tvec = ["<<X<<", "<<Y<<", "<<Z<<"] mm\n";
    cout<<"distance = "<<dist_cam_to_armor<<" mm\n";
}
// 计算pnp
bool runPnP(const vector<Point3d>& obj, const vector<Point2d>& img, int flag, const string& name, int armor_id)
{
    Mat rvec, tvec;
    bool ok= solvePnP(obj, img, matrix, dist, rvec, tvec, false, flag);
    cout<<"---- "<<name<<" ----\n";
    if(!ok){
        cout<<"solvePnP 失败\n";
        return false;
    }
    printPose(rvec, tvec, armor_id);
    double rmse= reprojRMSE(obj, img, rvec, tvec);
    cout<<"reproj RMSE = "<<rmse<<" px\n\n";
    return true;
}

int find_best_digit(const Rect& armor_box, const vector<DigitDet>& digits_this_frame) {
    if(digits_this_frame.empty()) return -1;
    Point ac= (armor_box.tl()+ armor_box.br())/2;
    double best_d= 1e18; int best_idx= -1;
    for(int i=0; i<digits_this_frame.size(); i++) {
        Point dc= (digits_this_frame[i].box.tl()+ digits_this_frame[i].box.br())/2;
        double dx= ac.x- dc.x, dy= ac.y- dc.y;
        double d2= dx*dx+ dy*dy;
        if(d2< best_d) { best_d= d2; best_idx= i; }
    }
    return best_idx;
}

void number_contours(Mat img_final, Mat img_resize, double fps) {
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(img_final, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  //vector<DigitDet> digits_this_frame; // ← 本帧识别到的数字列表vector<DigitDet> digits_this_frame; // ← 本帧识别到的数字列表
  for (int i = 0; i < contours.size(); i++) {
    double area = contourArea(contours[i]);
    if (area < 300) {
      continue;
    } else {
      Rect r = boundingRect(contours[i]); // 轮廓外接矩形
      // —— 推理部分开始：对每个 r 取 ROI，送入模型识别 —— //
      Rect rr = r & Rect(0, 0, img_resize.cols, img_resize.rows); // 防止越界
      if (rr.width > 15 && rr.height > 25) { // 过滤过小框
        Mat roi = img_resize(rr).clone();    // 取 ROI
        auto pred = infer_one(roi);          // (label,conf)

        // 写结果，预测的数字
        string txt = pred.first; //+ " "+ to_string(pred.second);
        Scalar color(0, 0, 0);
        if (txt == "3") {
          color = Scalar(255, 255, 0);
        } else if (txt == "4") {
          color = Scalar(0, 0, 255);
        }

        if (color != Scalar(0, 0, 0)) {
          putText(img_resize, txt, Point(rr.x, max(0, rr.y - 5)),
                  FONT_HERSHEY_SIMPLEX, 0.7, color, 2);    // 画预测文本
          rectangle(img_resize, r.tl(), r.br(), color, 2); // 画框
          circle(img_resize, (r.tl() + r.br()) / 2, 4,
                 Scalar(0, 255, 0)); // 画中心点
          DigitDet d; 
          d.box= rr; d.label= txt;  
          digits_this_frame.push_back(d);
        }
      }
    }
  }
  putText(img_resize, "fps:" + to_string(fps), Point(700, 50),
          FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
  imshow("img_final", img_resize);
}

void redcontours(Mat img_final, Mat img_resize) {
  int hmin = 0, smin = 204, vmin = 170; // 红色灯条的参数
  int hmax = 100, smax = 255, vmax = 255;
  Scalar lower(hmin, smin, vmin);
  Scalar upper(hmax, smax, vmax);
  inRange(img_final, lower, upper, img_final);
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(img_final, contours, hierarchy, RETR_EXTERNAL,
               CHAIN_APPROX_SIMPLE);

  for (int i = 0; i < contours.size(); i++) {
    double area = contourArea(contours[i]);

    Rect r = boundingRect(contours[i]);
    int k= find_best_digit(r, digits_this_frame);
    if(k<0) continue; // 没匹配到数字，不进行PnP
    const string& lab= digits_this_frame[k].label;
    // 只保留 0~9 的数字ID；若你的标签带字母，按需处理
    if(lab.empty() || !isdigit(lab[0])) continue;
    int armor_id= lab[0]- '0';

    vector<Point2d> light(6);
    light[0] = Point2d(r.x, r.y);                   
    light[1] = Point2d(r.x, r.y + r.height/2.0);    
    light[2] = Point2d(r.x, r.y + r.height);        
    light[3] = Point2d(r.x + r.width, r.y);                
    light[4] = Point2d(r.x + r.width, r.y + r.height/2.0); 
    light[5] = Point2d(r.x + r.width, r.y + r.height);
    vector<Point3d> obj= makeObjectPoints();
    runPnP(obj, light, SOLVEPNP_ITERATIVE, "SOLVEPNP_ITERATIVE", armor_id); // 迭代法，稳、精度高
    rectangle(img_resize, r.tl(), r.br(), Scalar(255, 255, 255), 2);
    circle(img_resize, (r.tl() + r.br()) / 2, 4, Scalar(255, 255, 255));
    putText(img_resize, "ID:"+ lab, r.tl()+ Point(0,-5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 2);
  }
  imshow("img_final", img_resize);
}


int main() {
  string path = "/home/liusi/文档/code/TDT-task4/Infantry_red.avi";
  VideoCapture cap(path);
  // ====== 加载 TorchScript 模型与标签 ======
  string model_path =
      "/home/liusi/文档/code/TDT-task4/model_ts.pt"; // 你的 TorchScript
  string label_path =
      "/home/liusi/文档/code/TDT-task4/p4_1_test.txt"; // 每行一个：0~9
  try {
    g_module = torch::jit::load(model_path); // 加载模型
    g_module.to(g_device);
    g_module.eval();
  } catch (const c10::Error &e) {
    cerr << "模型加载失败: " << e.what() << endl;
    return -1;
  }
  g_labels = load_labels(label_path);
  if (g_labels.empty()) {
    cerr << "标签文件为空或路径错误" << endl;
    return -1;
  }

  Mat img, img_hsv, hsv_mask, img_resize, img_gray, hsv_blur, gray_blur, img_clahe, final_mask;

  double fps = 0.0;
  int64 t1, t2;
  while (true) {
    t1 = getTickCount();
    cap.read(img);

    resize(img, img_resize, Size(900, 600));    // hsv部分
    cvtColor(img_resize, img_hsv, COLOR_BGR2HSV);
    GaussianBlur(img_hsv, hsv_blur, Size(5, 5), 5);
    redcontours(hsv_blur, img_resize);

    // 数字部分
    int hmin = 0, smin = 0, vmin = 64; // 数字，即装甲板的参数
    int hmax = 140, smax = 90, vmax = 255;
    // int hmin= 0, smin= 204, vmin= 170;//红色灯条的参数
    // int hmax= 100, smax= 255, vmax= 255;
    Scalar lower(hmin, smin, vmin);
    Scalar upper(hmax, smax, vmax);
    inRange(hsv_blur, lower, upper, hsv_mask);

    cvtColor(img_resize, img_gray, COLOR_BGR2GRAY);
    GaussianBlur(img_gray, gray_blur, Size(5, 5), 5);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(16, 16));
    clahe->apply(gray_blur, img_clahe); // 除噪点
    morphologyEx(img_clahe, img_clahe, MORPH_OPEN,
                 getStructuringElement(MORPH_RECT, Size(3, 3)));

    Mat bin = img_clahe;
    threshold(img_clahe, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    Mat labels, stats, centroids;
    int n = connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
    Mat begin_mask = Mat::zeros(bin.size(), CV_8U);
    for (int i = 1; i < n; i++) { // 0是背景
      int area = stats.at<int>(i, CC_STAT_AREA);
      int w = stats.at<int>(i, CC_STAT_WIDTH);
      int h = stats.at<int>(i, CC_STAT_HEIGHT);
      float ar = (float)h / (float)w; // 高宽比

      if (area >= 600 && area <= 2750 && ar > 0.25 &&
          ar < 3.75) { // 面积，高宽比
        begin_mask.setTo(255, labels == i);
      }
    }

    bitwise_and(hsv_mask, begin_mask, final_mask);
    number_contours(final_mask, img_resize, fps);
    
    if (waitKey(30) == 27) {
      break;
    }
    t2 = getTickCount();
    fps = getTickFrequency() / (t2 - t1);
  }
  return 0;
}