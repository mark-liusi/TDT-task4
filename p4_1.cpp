#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
void getcontours(Mat img_final, Mat img_resize, double fps)
{
   vector<vector<Point>> contours;
   vector<Vec4i> hierarchy;
   findContours(img_final, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
   
   for(int i= 0; i<contours.size(); i++){
      double area= contourArea(contours[i]);
      if(area<300){
         continue;
      }else{
         Rect r = boundingRect(contours[i]);
         rectangle(img_resize, r.tl(), r.br(), Scalar(0,255,0), 2);
         circle(img_resize, (r.tl()+r.br())/2, 4, Scalar(0, 255, 0));
      }
      
      //rectangle(img_resize, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
   }
   putText(img_resize, "fps:"+to_string(fps), Point(700, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
   imshow("img_final", img_resize);
}
void redcontours(Mat img_final, Mat img_resize){
   int hmin= 0, smin= 204, vmin= 170;//红色灯条的参数
   int hmax= 100, smax= 255, vmax= 255;
   Scalar lower(hmin, smin, vmin);
   Scalar upper(hmax, smax, vmax);
   inRange(img_final, lower, upper, img_final);
   vector<vector<Point>> contours;
   vector<Vec4i> hierarchy;
   findContours(img_final, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
   
   for(int i= 0; i<contours.size(); i++){
      double area= contourArea(contours[i]);
      
      Rect r = boundingRect(contours[i]);
      rectangle(img_resize, r.tl(), r.br(), (255,255,255), 2);
      circle(img_resize, (r.tl()+r.br())/2, 4, (255, 255, 255));
      
      //rectangle(img_resize, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
   }
   imshow("img_final", img_resize);
}
int main()
{
   string path= "Infantry_red.avi";
   VideoCapture cap(path);
   Mat img, img_hsv, hsv_mask, img_resize, img_gray, hsv_blur, gray_blur, img_clahe, final_mask;
   

   double fps= 0.0;
   int64 t1, t2;
   while(true){
      t1= getTickCount();
      cap.read(img);
      
      resize(img, img_resize, Size(900, 600));
      imshow("img", img_resize);

      //hsv部分
      cvtColor(img_resize, img_hsv, COLOR_BGR2HSV);
      GaussianBlur(img_hsv, hsv_blur, Size(5, 5), 5);
      redcontours(hsv_blur, img_resize);
      int hmin= 0, smin= 0, vmin= 64;//数字，即装甲板的参数
      int hmax= 140, smax= 90, vmax= 255;
      // int hmin= 0, smin= 204, vmin= 170;//红色灯条的参数
      // int hmax= 100, smax= 255, vmax= 255;
      Scalar lower(hmin, smin, vmin);
      Scalar upper(hmax, smax, vmax);
      inRange(hsv_blur, lower, upper, hsv_mask);
      //imshow("hsv_mask", hsv_mask);
      //getcontours(hsv_mask, img_resize);
      
      //inRange(hsv_blur, lower, upper, hsv_mask);
      //getcontours(hsv_mask, img_resize);
      //imshow("img hsv", hsv_mask);


      cvtColor(img_resize, img_gray, COLOR_BGR2GRAY);
      GaussianBlur(img_gray, gray_blur, Size(5, 5), 5);
      Ptr<CLAHE> clahe= createCLAHE(2.0, Size(16, 16));
      clahe->apply(gray_blur, img_clahe);//除噪点
      morphologyEx(img_clahe, img_clahe, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3,3)));


      Mat bin= img_clahe;
      threshold(img_clahe, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
      Mat labels, stats, centroids;
      int n = connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
      Mat begin_mask = Mat::zeros(bin.size(), CV_8U);
      for(int i=1; i<n; i++){  //0是背景
         int area = stats.at<int>(i, CC_STAT_AREA);
         int w= stats.at<int>(i, CC_STAT_WIDTH);
         int h    = stats.at<int>(i, CC_STAT_HEIGHT);
         float ar = (float)h / (float)w; //高宽比

         if(area>= 600 &&area<= 2750 && ar> 0.25 &&ar< 3.75){//面积，高宽比
            begin_mask.setTo(255, labels == i);
         }
      }

      bitwise_and(hsv_mask, begin_mask, final_mask);
      //imshow("img_gray", begin_mask);
      getcontours(final_mask, img_resize, fps);
      //imshow("final_mask", final_mask);
      
     
      //imshow("begin_mask", begin_mask);
      
      
      if(waitKey(30)== 27){
         break;
      }
      t2= getTickCount();
      fps= getTickFrequency()/(t2 - t1) ;
   }
   return 0;
}


