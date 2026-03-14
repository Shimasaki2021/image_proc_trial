#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 画像を作成 (黒背景)
    cv::Mat image = cv::Mat::zeros(300, 300, CV_8UC3);
    // 文字を描画
    cv::putText(image, "Hello WSL2 OpenCV", cv::Point(30, 150), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    
    // ウィンドウに表示
    cv::imshow("Sample Window", image);
    cv::waitKey(0);
    return 0;
}
