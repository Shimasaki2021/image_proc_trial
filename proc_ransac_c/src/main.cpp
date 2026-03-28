#include <opencv2/opencv.hpp>
#include <iostream>

double calculateMedian(const cv::Mat& src) 
{
    cv::Mat flat;
    src.reshape(1, 1).copyTo(flat); // 1次元配列に変換
    std::vector<int> vec;
    flat.copyTo(vec);
    std::sort(vec.begin(), vec.end()); // ソート
    return vec[vec.size() / 2]; // 中央値を取得
}

void extractEdge(const cv::Mat &img_in_g, cv::Mat &img_edge)
{
    // https://qiita.com/kotai2003/items/662c33c15915f2a8517e
    cv::Scalar mean_val, sigma_val;
    double med_val, sigma;
    int min_val, max_val;

    med_val = calculateMedian(img_in_g);
    cv::meanStdDev(img_in_g, mean_val, sigma_val);
    sigma = sigma_val[0] / 255.0;
    min_val = (int)std::max(0.0,   (1.0 - sigma) * med_val);
    max_val = (int)std::min(255.0, (1.0 + sigma) * med_val);

    cv::Canny(img_in_g, img_edge, min_val, max_val);
    printf("img_out(%d %d) = cv.Canny(img_in_g, img_edge, %d, %d)\n", 
        img_edge.cols, img_edge.rows, min_val, max_val);
    return;
}

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
        const char *p_img_file;
        cv::Mat img_in, img_in_g, img_edge;

        p_img_file = argv[1];
        img_in = cv::imread(p_img_file);

        // エッジ検出
        cv::cvtColor(img_in, img_in_g, cv::COLOR_BGR2GRAY);
        extractEdge(img_in_g, img_edge);

        // ウィンドウに表示
        // cv::imshow("Sample Window", img_in);
        cv::imshow("Sample Window", img_edge);
        cv::waitKey(0);

        ret = 0;
    }

    return ret;
}
