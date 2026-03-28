#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[]) 
{
    int ret;

    if(argc < 2)
    {
        printf("Usage: %s [img file path]\n", argv[0]);
        ret = -1;
    }
    else
    {
        // 画像を作成
        const char *p_img_file = argv[1];
        cv::Mat image = cv::imread(p_img_file);

        // 文字を描画
        cv::putText(image, "Hello WSL2 OpenCV", cv::Point(30, 150), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        // ウィンドウに表示
        cv::imshow("Sample Window", image);
        cv::waitKey(0);

        ret = 0;
    }

    return ret;
}
