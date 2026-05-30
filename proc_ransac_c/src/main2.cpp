#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <filesystem>
#include <iomanip>
#include <algorithm>

#include "main.hpp"
#include "fig.hpp"
#include "debug.hpp"

std::pair<std::vector<int>, int> nmSuppression(
    const std::vector<CvRect>& boxes,
    const std::vector<int>& scores,
    float iou_th = 0.45f,
    int top_k = 200) 
{
    int N;

    N = boxes.size();

    if (N == 0) 
    {
        return {{}, 0};
    }

    std::vector<int> keep(N, 0);
    int count;
    std::vector<float> x1(N), y1(N), x2(N), y2(N), area(N);
    std::vector<int> idx(N);

    for (int i = 0; i < N; i++) 
    {
        x1[i] = (float)boxes[i].pt_min_.x;
        y1[i] = (float)boxes[i].pt_min_.y;
        x2[i] = (float)boxes[i].pt_max_.x;
        y2[i] = (float)boxes[i].pt_max_.y;
        area[i] = (x2[i] - x1[i]) * (y2[i] - y1[i]);
    }

    for (int i = 0; i < N; i++) 
    {
        idx[i] = i;
    }

    std::sort(idx.begin(), 
            idx.end(), 
            [&](int a, int b) 
            {
                return scores[a] < scores[b];
            }
    );

    if((int)idx.size() > top_k) 
    {
        idx.erase(idx.begin(), idx.end() - top_k);
    }

    count = 0;

    while (!idx.empty()) 
    {
        int i;
        std::vector<int> new_idx;

        i = idx.back();
        idx.pop_back();

        keep[count++] = i;

        if (idx.empty()) 
        {
            break;
        }

        new_idx.reserve(idx.size());

        for (int j : idx) 
        {
            float xx1 = std::max(x1[j], x1[i]);
            float yy1 = std::max(y1[j], y1[i]);
            float xx2 = std::min(x2[j], x2[i]);
            float yy2 = std::min(y2[j], y2[i]);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);

            float inter = w * h;
            float uni = area[j] + area[i] - inter;
            float IoU = inter / uni;

            if (IoU <= iou_th) 
            {
                new_idx.push_back(j);
            }
        }

        idx.swap(new_idx);
    }

    keep.resize(count);

    return {keep, count};
}

void extractObjectRANSAC(const CvPointList &edge_pixels, 
                         FigType &obj_type, 
                         FigList &det_objs_sup, 
                         CfgType &cfg)
{
    FigList det_objs;
    std::shared_ptr<Fig> target_fig;
    int num_iter;
    int count_iter;
    // int num_max_inlier;
    //int num_inlier;

    det_objs.clear();

    if(obj_type.figtype_ == FigType::FIGTYPE_LINE_)
    {
        target_fig = std::make_shared<FigLine>(cfg);
        // det_obj    = std::make_shared<FigLine>(cfg);
    }
    else
    {
        target_fig = std::make_shared<FigCircle>(cfg);
        // det_obj    = std::make_shared<FigCircle>(cfg);
    }

    num_iter = (int)((float)edge_pixels.size() * strtof(cfg["RANSAC_NUM_ITER_PER_EDGE"].c_str(), NULL));

    // num_max_inlier = 0;
    // (*det_obj) = target_fig;
    count_iter = 0;

    while(count_iter < num_iter)
    {
        CvPointList choise_pixels;
        bool is_valid;

        target_fig->reset();

        // エッジ点群から、直線／円の作成に必要な点（直線なら2点、円なら3点）をランダムに抽出
        target_fig->choiseRandomPixels(edge_pixels, choise_pixels);

        if(target_fig->isEnableCreate(choise_pixels) == true)
        {
            // 抽出した点から直線／円を作成
            target_fig->create(choise_pixels);

            // 作成した直線／円周上の点の数（inlier）をカウント
            target_fig->countInlier(edge_pixels, target_fig->dist_th_);
            // num_inlier = target_fig->countInlier(edge_pixels, target_fig->dist_th_);
            
            // inlier点群の特徴抽出（外接矩形）、フィルタリング
            target_fig->calcInlierBBox();
            is_valid = target_fig->filteredByInlierPixels();

#if 1
            if(is_valid == true)
            {
                det_objs.push_back(target_fig->clone());
            }
#else
            if(num_inlier > num_max_inlier)
            {
                // inlier数最大の直線／円を返す
                num_max_inlier = num_inlier;
                (*det_obj) = target_fig;
            }
#endif

            count_iter++;
        }
    }

    // 外接矩形が重複するものは、一番inlier数が多いもののみ残す（Non-maximum supression）
    std::vector<CvRect> boxes;
    std::vector<int> scores;
    std::vector<int> sup_idx;

    for(auto det_obj : det_objs)
    {
        boxes.push_back(det_obj->inlier_bbox_);
        scores.push_back(det_obj->num_inlier_);
    }
    auto sup_res = nmSuppression(boxes, scores, 0.1);
    sup_idx = sup_res.first;

    det_objs_sup.clear();

    for(int idx : sup_idx)
    {
        det_objs_sup.push_back(det_objs[idx]->clone());
    }

    return;
}

void extractObjects(cv::Mat &img_edge, 
                    FigList &det_objs_all, 
                    CfgType &cfg,
                    DebugOut &dbg)
{
    FigType target_obj_type;
    CvPointList edge_pixels;

    det_objs_all.clear();

    while(false == target_obj_type.isNone())
    {
        FigList det_objs;
        int len_edge_pixels;
        std::shared_ptr<Fig> det_obj(nullptr);
        //char fname_img_edge[128];

        // エッジ画像からエッジ点群を抽出
        cv::findNonZero(img_edge, edge_pixels);
        len_edge_pixels = edge_pixels.size();

        if(len_edge_pixels <= 0)
        {
            break;
        }

        // エッジ点群から直線／円を検出
        extractObjectRANSAC(edge_pixels, target_obj_type, det_objs, cfg);

#if 1
        if(det_objs.size() > 0)
        {
            // [検出できた場合] 
            det_objs_all.insert(det_objs_all.end(), det_objs.begin(), det_objs.end());

            // 検出した直線／円に含まれるエッジ点(inlier点)を削除
            for(auto det_obj : det_objs)
            {
                det_obj->erasePixels(img_edge);
            }

            dbg.printLogLine("[%s] detect.", target_obj_type.toString().c_str());
            dbg.dumpImg(img_edge, std::string("edge_tmp_after_") + target_obj_type.toString());
        }

        // 次の種別の検出図形へ
        target_obj_type.next();
#else
        if((det_obj != nullptr) && (det_obj->is_valid_ == true))
        {
            // [検出できた場合] 
            det_objs.push_back(det_obj);

            // 検出した直線／円に含まれるエッジ点(inlier点)を削除し、
            // 同じ種別の図形検出を継続
            det_obj->erasePixels(img_edge);

            dbg.printLogLine("[%lu] detect %s",
                             det_objs.size(), 
                             target_obj_type.toString().c_str());
            dbg.printLogLine("  %s", det_obj->toString().c_str());
            
            snprintf(fname_img_edge, sizeof(fname_img_edge), "edge_tmp%lu_%s",
                     det_objs.size(), target_obj_type.toString().c_str());
            dbg.dumpImg(img_edge, std::string(fname_img_edge));
        }
        else
        {
            // [検出できなかった場合] 次の種別の検出図形へ
            target_obj_type.next();
        }
#endif
    }

    return;
}

double calculateMedian(const cv::Mat& src) 
{
    cv::Mat flat;
    src.reshape(1, 1).copyTo(flat); // 1次元配列に変換
    std::vector<int> vec;
    flat.copyTo(vec);
    std::sort(vec.begin(), vec.end()); // ソート

    return vec[vec.size() / 2]; // 中央値を取得
}

void extractEdge(const cv::Mat &img_in_g, cv::Mat &img_edge, DebugOut &dbg)
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

    dbg.printLogLine("img_out(%d %d) = cv.Canny(img_in_g, img_edge, %d, %d)", 
        img_edge.cols, img_edge.rows, min_val, max_val);

    return;
}

int main(int argc, char *argv[]) 
{
    CfgType cfg = 
    {
        // RANSAC繰り返し回数（エッジ点数に対する倍率を指定）
        // {"RANSAC_NUM_ITER_PER_EDGE", "1.5"},
        {"RANSAC_NUM_ITER_PER_EDGE", "4.5"},

        // 検出図形（直線or円）との距離閾値(inlier閾値)[pixel]
        {"INLIER_DIST_TH", "1.0"},

        // inlier点群の数の下限閾値[pixel]
        {"INLIER_NUM_MIN_TH", "10"},

        // inlier点群の密度(0～1)閾値
        {"INLIER_LINE_DENSE_TH", "0.5"},    // 直線
        {"INLIER_CIRCLE_DENSE_TH", "0.5"},  // 円

        // 線分の最小長[pixel]
        {"LINE_MIN_LEN_TH", "20"},
        // 円の最小半径[pixel]
        {"CIRCLE_MIN_R_TH", "5"},

        // 出力ディレクトリ
        {"OUTPUT_DIR", "output_cpp2"},
    };
    int ret;
    

    if(argc < 2)
    {
        printf("Usage: %s [img file path]\n", argv[0]);
        ret = -1;
    }
    else
    {
        const char *img_fpath;
        cv::Mat img_in, img_in_g, img_edge;
        std::filesystem::path fpath;
        std::string img_fname, img_fname_base;
        FigList det_objs;
        std::chrono::system_clock::time_point time_s, time_e;
        double time_elapsed;

        img_fpath = argv[1];
        img_in = cv::imread(img_fpath);

        fpath = std::filesystem::path(img_fpath);
        img_fname = fpath.filename().string();
        img_fname_base = fpath.stem().string();

        DebugOut dbg(cfg["OUTPUT_DIR"].c_str(), img_fname_base.c_str());
        dbg.is_out_ = true;
        dbg.openLogFile("log.txt");

        time_s = std::chrono::system_clock::now();

        // エッジ検出
        cv::cvtColor(img_in, img_in_g, cv::COLOR_BGR2GRAY);
        extractEdge(img_in_g, img_edge, dbg);
        dbg.dumpImg(img_edge, "edge");

        // 直線／円検出
        extractObjects(img_edge, det_objs, cfg, dbg);

        time_e = std::chrono::system_clock::now();
        time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_e - time_s).count(); 

        // 検出結果を重畳描画
        for(auto det_obj : det_objs)
        {
            det_obj->draw(img_in);
        }

        dbg.dumpImg(img_in, "det");
        dbg.printLogLine("time[sec] = %f", time_elapsed/1000.0);

        dbg.closeLogFile();

        // ウィンドウに表示
        // cv::imshow("Sample Window", img_in);
        // cv::waitKey(0);

        ret = 0;
    }

    return ret;
}
