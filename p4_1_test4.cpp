// main.cpp
#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include <ATen/ATen.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp> // KalmanFilter

using namespace std;
using namespace cv;

// ===================== 基础结构体 =====================
struct Armor {
  int id = -1;
  Mat rvec, tvec;
  double distance = 0; // mm
  double reproj = 0;
};
struct DigitDet {
  Rect box;
  string label;
};

// ===================== 全局态 =====================
static vector<DigitDet> g_digits_this_frame;
static unordered_map<int, Armor> g_armor_map; // id -> armor pose (for distance)

// 相机内参
Mat matrix = (Mat_<double>(3, 3) << 1777.4091, 0, 710.7598, 0, 1775.4171,
              534.7207, 0, 0, 1);
Mat dist =
    (Mat_<double>(1, 5) << -0.563142, 0.183051, 0.001964, 0.000925, 0.568833);

// ===================== 2D Kalman (u,v, du, dv) 用像素域做提前量
// =====================
struct KF4 {
  KalmanFilter kf;
  bool inited = false;
  chrono::steady_clock::time_point last_tp;
  KF4() : kf(4, 2, 0, CV_32F) {}
};
static unordered_map<int, KF4> g_kf2d_map;

static void kf2d_init(KF4 &w, float u0, float v0) {
  w.kf = KalmanFilter(4, 2, 0, CV_32F);
  // 状态: [u, v, du, dv]^T
  // 测量: [u, v]^T
  // 初始 F 用 dt=1，占位，实际每帧会改
  w.kf.transitionMatrix =
      (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
  setIdentity(w.kf.measurementMatrix);              // H
  setIdentity(w.kf.processNoiseCov, Scalar(1e-2));  // Q（可调大/小）
  setIdentity(w.kf.measurementNoiseCov, Scalar(2)); // R（像素测量噪声）
  setIdentity(w.kf.errorCovPost, Scalar(1000));     // P0

  w.kf.statePost = (Mat_<float>(4, 1) << u0, v0, 0.f, 0.f);
  w.inited = true;
  w.last_tp = chrono::steady_clock::now();
}

static void kf2d_setF(KF4 &w, double dt) {
  // 给速度一点阻尼，防直线飞：drag^dt
  float drag = pow(0.98f, (float)max(0.0, dt));
  w.kf.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, (float)dt, 0, 0, 1, 0,
                           (float)dt, 0, 0, drag, 0, 0, 0, 0, drag);
}

// ===================== 物理点（用于距离，PnP 4角） =====================
static vector<Point3d> makeObjectPoints4() {
  double W = 135.0, H = 55.0; // mm
  double w = W / 2, h = H / 2;
  return {{-w, -h, 0}, {w, -h, 0}, {w, h, 0}, {-w, h, 0}};
}

// ===================== Torch 推理相关 =====================
static torch::jit::script::Module g_module;
static vector<string> g_labels;
static int g_in_w = 28, g_in_h = 28;
static bool g_to_gray = true;
static vector<double> g_mean = {0.1307};
static vector<double> g_std = {0.3081};
static torch::Device g_device = torch::kCPU;

static vector<string> load_labels(const string &path) {
  vector<string> labels;
  ifstream fin(path);
  string line;
  while (getline(fin, line))
    if (!line.empty())
      labels.push_back(line);
  return labels;
}
static torch::Tensor preprocess(Mat roi) {
  Mat img_resized, img_conv;
  resize(roi, img_resized, Size(g_in_w, g_in_h));
  if (g_to_gray)
    cvtColor(img_resized, img_conv, COLOR_BGR2GRAY);
  else
    cvtColor(img_resized, img_conv, COLOR_BGR2RGB);
  img_conv.convertTo(img_conv, CV_32F, 1.0 / 255.0);
  torch::Tensor t =
      torch::from_blob(img_conv.data, {img_conv.rows, img_conv.cols},
                       torch::kFloat32)
          .clone();
  t = t.unsqueeze(0);
  t = (t - (float)g_mean[0]) / (float)g_std[0];
  t = t.unsqueeze(0);
  return t.to(g_device);
}
static pair<string, float> infer_one(Mat roi) {
  torch::InferenceMode no_grad;
  torch::Tensor tin = preprocess(roi);
  vector<torch::jit::IValue> inputs = {tin};
  torch::Tensor logits = g_module.forward(inputs).toTensor();
  int pred_idx = logits.argmax(1).item<int>();
  torch::Tensor probs = torch::softmax(logits, 1);
  float conf = probs[0][pred_idx].item<float>();
  string label = (pred_idx >= 0 && pred_idx < (int)g_labels.size())
                     ? g_labels[pred_idx]
                     : "NA";
  return {label, conf};
}

// ===================== PnP / 辅助 =====================
static double reprojRMSE(const vector<Point3d> &obj, const vector<Point2d> &img,
                         const Mat &rvec, const Mat &tvec) {
  vector<Point2d> proj;
  projectPoints(obj, rvec, tvec, matrix, dist, proj);
  double se = 0.0;
  for (size_t i = 0; i < proj.size(); i++) {
    double dx = proj[i].x - img[i].x, dy = proj[i].y - img[i].y;
    se += dx * dx + dy * dy;
  }
  return sqrt(se / std::max(1.0, (double)proj.size()));
}
static bool runPnP(const vector<Point3d> &obj, const vector<Point2d> &img,
                   int flag, const string &name, int armor_id) {
  Mat rvec, tvec;
  bool ok = solvePnP(obj, img, matrix, dist, rvec, tvec, false, flag);
  cout << "---- " << name << " ----\n";
  if (!ok) {
    cout << "solvePnP 失败\n";
    return false;
  }
  double X = tvec.at<double>(0), Y = tvec.at<double>(1), Z = tvec.at<double>(2);
  double dist_cam_to_armor = sqrt(X * X + Y * Y + Z * Z);
  Armor &A = g_armor_map[armor_id];
  A.id = armor_id;
  A.rvec = rvec.clone();
  A.tvec = tvec.clone();
  A.distance = dist_cam_to_armor;
  cout << "tvec = [" << X << "," << Y << "," << Z
       << "] mm, dist=" << dist_cam_to_armor << " mm\n";
  cout << "reproj RMSE = " << reprojRMSE(obj, img, rvec, tvec) << " px\n\n";
  return true;
}
static inline int clampi(int v, int lo, int hi) {
  return std::max(lo, std::min(hi, v));
}

// ===================== 工具：数字匹配到装甲框 =====================
static int find_best_digit(const Rect &armor_box, const vector<DigitDet> &ds) {
  if (ds.empty())
    return -1;
  Point ac = (armor_box.tl() + armor_box.br()) / 2;
  double best_d = 1e18;
  int best_idx = -1;
  for (int i = 0; i < (int)ds.size(); i++) {
    Point dc = (ds[i].box.tl() + ds[i].box.br()) / 2;
    double dx = ac.x - dc.x, dy = ac.y - dc.y;
    double d2 = dx * dx + dy * dy;
    if (d2 < best_d) {
      best_d = d2;
      best_idx = i;
    }
  }
  return best_idx;
}

// ===================== 数字分支（先做） =====================
static void number_contours(Mat img_final, Mat &img_show, double fps) {
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(img_final, contours, hierarchy, RETR_EXTERNAL,
               CHAIN_APPROX_SIMPLE);

  for (int i = 0; i < (int)contours.size(); i++) {
    double area = contourArea(contours[i]);
    if (area < 300)
      continue;

    Rect r = boundingRect(contours[i]);
    Rect rr = r & Rect(0, 0, img_show.cols, img_show.rows);
    if (rr.width <= 15 || rr.height <= 25)
      continue;

    Mat roi = img_show(rr).clone();
    auto pred = infer_one(roi);
    string txt = pred.first;

    if (!txt.empty()) {
      Scalar color(0, 255, 0);
      putText(img_show, txt, Point(rr.x, max(0, rr.y - 5)),
              FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
      rectangle(img_show, rr, color, 1);

      // 关键：画出“数字框的中心点”（绿色实心）——这就是你要的中心
      Point center = (rr.tl() + rr.br()) / 2;
      circle(img_show, center, 4, Scalar(0, 255, 0), -1);

      g_digits_this_frame.push_back({rr, txt});
    }
  }
  putText(img_show, "fps:" + to_string(fps), Point(700, 50),
          FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
}

// ===================== 红灯条分支（后做）：最大目标 + PnP(取距离) + 2D KF
// 提前量 =====================
static const double BULLET_SPEED_MM_S = 250.0; // 25 cm/s
static const double TOF_MAX_SEC = 0.30;        // 防超长外推

static void redcontours(Mat img_hsv_or_blur, Mat &img_show) {
  // 阈值：红灯条（按你原参数，可再调）
  int hmin = 0, smin = 204, vmin = 170;
  int hmax = 100, smax = 255, vmax = 255;
  Mat mask;
  inRange(img_hsv_or_blur, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax),
          mask);

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    putText(img_show, "No target", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
            Scalar(0, 255, 255), 2);
    return;
  }

  // 修改：遍历所有轮廓，而不是只选最大的
  for (int i = 0; i < (int)contours.size(); i++) {
    double area = contourArea(contours[i]);

    Rect r = boundingRect(contours[i]);

    // 配最近数字作为 armor_id
    int k = find_best_digit(r, g_digits_this_frame);
    if (k < 0)
      continue; // 没匹配到数字，跳过这个轮廓

    const string &lab = g_digits_this_frame[k].label;
    if (lab.empty() || !isdigit(lab[0]))
      continue;
    int armor_id = lab[0] - '0';

    // PnP 取距离（4角）
    vector<Point2d> img4 = {Point2d(r.x, r.y), Point2d(r.x + r.width, r.y),
                            Point2d(r.x + r.width, r.y + r.height),
                            Point2d(r.x, r.y + r.height)};
    vector<Point3d> obj4 = makeObjectPoints4();
    bool ok =
        runPnP(obj4, img4, SOLVEPNP_ITERATIVE, "SOLVEPNP_ITERATIVE", armor_id);

    // 高亮当前“待击打”装甲框
    rectangle(img_show, r, Scalar(255, 0, 255), 1);
    putText(img_show, "ID:" + lab, r.tl() + Point(0, -5), FONT_HERSHEY_SIMPLEX,
            0.8, Scalar(255, 255, 255), 2);

    if (!ok)
      continue;

    // 用“数字框中心”（绿色点）作为当前测量中心
    Point dig_center =
        (g_digits_this_frame[k].box.tl() + g_digits_this_frame[k].box.br()) / 2;
    // 再次强调一下它（大一点）
    circle(img_show, dig_center, 5, Scalar(0, 255, 0), -1);

    // 2D KF（像素域）更新与预测
    KF4 &W = g_kf2d_map[armor_id];
    auto now_tp = chrono::steady_clock::now();
    double dt = 0.0;
    if (!W.inited) {
      kf2d_init(W, (float)dig_center.x, (float)dig_center.y);
      dt = 0.0;
    } else {
      dt = chrono::duration<double>(now_tp - W.last_tp).count();
      kf2d_setF(W, dt);
    }
    W.last_tp = now_tp;

    // predict + correct
    Mat z = (Mat_<float>(2, 1) << (float)dig_center.x, (float)dig_center.y);
    W.kf.predict();
    Mat xpost = W.kf.correct(z);

    float u = xpost.at<float>(0), v = xpost.at<float>(1);
    float du = xpost.at<float>(2), dv = xpost.at<float>(3);

    // 飞行时间（用 PnP 距离）
    Armor &A = g_armor_map[armor_id];
    double tof = (A.distance > 0) ? (A.distance / BULLET_SPEED_MM_S) : 0.0;
    if (tof > TOF_MAX_SEC)
      tof = TOF_MAX_SEC;
    if (tof < 0.0)
      tof = 0.0;

    // 像素域提前量（常速度）：hit = (u, v) + (du, dv)*tof
    int hit_u = clampi((int)lround(u + du * tof), 0, img_show.cols - 1);
    int hit_v = clampi((int)lround(v + dv * tof), 0, img_show.rows - 1);

    // 可视化：绿色=当前中心（数字框），红色=命中预测点
    circle(img_show, Point((int)lround(u), (int)lround(v)), 6,
           Scalar(0, 255, 0), -1);
    circle(img_show, Point(hit_u, hit_v), 10, Scalar(0, 0, 255), 2);
    line(img_show, Point((int)lround(u), (int)lround(v)), Point(hit_u, hit_v),
         Scalar(0, 255, 255), 1);

    char info[128];
    snprintf(info, sizeof(info), "TOF=%.3fs dist=%.0fmm", tof, A.distance);
    putText(img_show, info, r.tl() + Point(0, -25), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(255, 255, 255), 2);
  }
}

// ===================== 主函数 =====================
int main() {
  // 视频路径
  string path = "/home/liusi/文档/code/TDT-task4/Infantry_red.avi";
  VideoCapture cap(path);
  if (!cap.isOpened()) {
    cerr << "无法打开视频: " << path << endl;
    return -1;
  }

  // 播放控制
  double src_fps = cap.get(CAP_PROP_FPS);
  if (!(src_fps > 0) || src_fps > 240)
    src_fps = 30.0;
  double playback_factor = 0.50; // 初始半速
  cout << "Source FPS=" << src_fps << ", playback x" << playback_factor << endl;

  // 模型与标签
  string model_path = "/home/liusi/文档/code/TDT-task4/model_ts.pt";
  string label_path = "/home/liusi/文档/code/TDT-task4/p4_1_test.txt";
  try {
    g_module = torch::jit::load(model_path);
    g_module.to(torch::kCPU);
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

  Mat img, img_hsv, hsv_mask, img_resize, img_gray, hsv_blur, gray_blur,
      img_clahe, final_mask;
  double fps = 0.0;
  int64 t1 = 0, t2 = 0;

  while (true) {
    auto frame_begin = chrono::steady_clock::now();
    t1 = getTickCount();
    if (!cap.read(img))
      break;

    // ===== 先：数字分支（本帧识别 & 画绿色中心）=====
    g_digits_this_frame.clear();
    resize(img, img_resize, Size(900, 600));
    cvtColor(img_resize, img_hsv, COLOR_BGR2HSV);
    GaussianBlur(img_hsv, hsv_blur, Size(5, 5), 5);

    // 数字粗分割（你原参数）
    int hmin = 0, smin = 0, vmin = 64;
    int hmax = 140, smax = 90, vmax = 255;
    inRange(hsv_blur, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax),
            hsv_mask);

    cvtColor(img_resize, img_gray, COLOR_BGR2GRAY);
    GaussianBlur(img_gray, gray_blur, Size(5, 5), 5);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(16, 16));
    clahe->apply(img_gray, img_clahe);
    morphologyEx(img_clahe, img_clahe, MORPH_OPEN,
                 getStructuringElement(MORPH_RECT, Size(3, 3)));
    Mat bin = img_clahe;
    threshold(img_clahe, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

    Mat labels, stats, centroids;
    int n =
        connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
    Mat begin_mask = Mat::zeros(bin.size(), CV_8U);
    for (int i = 1; i < n; i++) {
      int area = stats.at<int>(i, CC_STAT_AREA);
      int w = stats.at<int>(i, CC_STAT_WIDTH);
      int h = stats.at<int>(i, CC_STAT_HEIGHT);
      float ar = (float)h / (float)w;
      if (area >= 600 && area <= 2750 && ar > 0.25 && ar < 3.75)
        begin_mask.setTo(255, labels == i);
    }
    bitwise_and(hsv_mask, begin_mask, final_mask);
    number_contours(final_mask, img_resize, fps); // 这里面会画绿色点

    // ===== 后：红灯条（最大框）+ PnP(距) + 2D KF 像素域提前量 =====
    redcontours(hsv_blur, img_resize);

    // HUD：当前播放倍速
    char sp[64];
    snprintf(sp, sizeof(sp), "play x%.2f", playback_factor);
    putText(img_resize, sp, Point(10, img_resize.rows - 20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    imshow("armor_predict", img_resize);

    t2 = getTickCount();
    fps = getTickFrequency() / (t2 - t1);

    // 自适应播放延时
    auto frame_end = chrono::steady_clock::now();
    double used_ms =
        chrono::duration<double, std::milli>(frame_end - frame_begin).count();
    double target_ms = (1000.0 / src_fps) / playback_factor;
    int delay_ms = (int)max(1.0, target_ms - used_ms);

    int key = waitKey(delay_ms) & 0xFF;
    if (key == 27)
      break;
    else if (key == ']')
      playback_factor = min(4.0, playback_factor * 1.25);
    else if (key == '[')
      playback_factor = max(0.05, playback_factor / 1.25);
  }

  cap.release();
  destroyAllWindows();
  return 0;
}