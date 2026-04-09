
#include "fig.hpp"

// === Fig ===

Fig& Fig::operator=(const Fig& f)
{
    is_valid_        = f.is_valid_;
    num_inlier_      = f.num_inlier_;
    inlier_pixels_   = f.inlier_pixels_;
    inlier_dense_th_ = f.inlier_dense_th_;
    dist_th_         = f.dist_th_;
    min_inlier_th_   = f.min_inlier_th_;

    return (*this);
}

void Fig::operator=(const std::shared_ptr<Fig> &p)
{
    if(p != nullptr)
    {
        Fig::operator=(*p);
    }
    return;
}

void Fig::erasePixels(cv::Mat &img) const
{
    if((is_valid_ == true) && (inlier_pixels_.size() > 0))
    {
        // inlier点を削除(0塗りつぶし)する
        for(auto &px : inlier_pixels_)
        {
            img.at<uchar>(px.y, px.x) = 0;
        }
    }

    return;
}

std::string Fig::toString(void) const
{
    std::stringstream ss;
    ss << "{valid=" << is_valid_ << ",num_inlier=" << num_inlier_ << ",";
    return ss.str();
}

// === FigLine ===

FigLine& FigLine::operator=(const FigLine& f)
{
    Fig::operator=(f);

    a_ = f.a_;
    b_ = f.b_;
    c_ = f.c_;
    sqrt_a2_plus_b2_ = f.sqrt_a2_plus_b2_;
    inlier_bbox_min_ = f.inlier_bbox_min_;
    inlier_bbox_max_ = f.inlier_bbox_max_;
    len_lineseg_     = f.len_lineseg_;

    return (*this);
}

void FigLine::operator=(const std::shared_ptr<Fig> &p)
{
    const std::shared_ptr<FigLine> pline = std::dynamic_pointer_cast<FigLine>(p);

    if(pline != nullptr)
    {
        FigLine::operator=(*pline);
    }
    return;
}

void FigLine::choiseRandomPixels(const CvPointList &pixels, CvPointList &sel_pixels) const
{
    size_t num_pixels;

    num_pixels = pixels.size();

    sel_pixels.clear();

    if(num_pixels >= 2)
    {
        // pixelsの中からランダムに2点を選ぶ（重複禁止）
        std::random_device rd;
        std::mt19937 rand_gen;
        std::uniform_int_distribution<size_t> dist(0, num_pixels-1);
        size_t idx1, idx2;
        cv::Point px1, px2;

        rand_gen = std::mt19937(rd());

        idx1 = dist(rand_gen);
        do {
            idx2 = dist(rand_gen);
        } while (idx1 == idx2); // 同じ点が選ばれたらやり直し

        sel_pixels.push_back(pixels[idx1]);
        sel_pixels.push_back(pixels[idx2]);
    }

    return;
}

bool FigLine::isEnableCreate(const CvPointList &sel_pixels) const
{
    bool is_enable;

    is_enable = false;

    if(sel_pixels.size() >= 2)
    {
        if(    (sel_pixels[0].x != sel_pixels[1].x)
            || (sel_pixels[0].y != sel_pixels[1].y))
        {
            // [入力2点が異なる] 直線作成可
            is_enable = true;
        }
    }
    return is_enable;
}

void FigLine::create(const CvPointList &sel_pixels) 
{
    if(sel_pixels.size() >= 2)
    {
        double x0,y0;
        double x1,y1;
        double a,b,c;
        double sqrt_a2_plus_b2;

        x0 = (double)sel_pixels[0].x;
        y0 = (double)sel_pixels[0].y;
        x1 = (double)sel_pixels[1].x;
        y1 = (double)sel_pixels[1].y;

        // 直線のパラメータa,b,cを算出
        //   直線の方向ベクトル＝(x1​−x0​, y1​−y0​)
        //    → 直線の法線ベクトル＝(y0​−y1​,x1​−x0​)＝パラメータ(a,b)
        a = y0 - y1;
        b = x1 - x0;
        c = -(a * x0 + b * y0);

        sqrt_a2_plus_b2 = sqrt(a*a + b*b);

        if(sqrt_a2_plus_b2 > 1e-5)
        {
            a_ = a;
            b_ = b;
            c_ = c;
            sqrt_a2_plus_b2_ = sqrt_a2_plus_b2;
            is_valid_ = true;
        }
    }

    return;
}

int FigLine::countInlier(const CvPointList &pixels, double dist_th)
{
    num_inlier_ = 0;
    inlier_pixels_.clear();

    if(is_valid_ == true)
    {
        double dist;

        // 点と直線の距離 < 閾値 を満たす点の数をカウント
        for(const auto &px : pixels)
        {
            dist = std::abs(a_ * (double)px.x + b_ * (double)px.y + c_) / sqrt_a2_plus_b2_;

            if(dist < dist_th)
            {
                num_inlier_++;
                inlier_pixels_.push_back(px);
            }
        }

        if(num_inlier_ > min_inlier_th_)
        {
            // inlier点群の外接矩形/線分長を算出(近似)
            calcInlierBBox(inlier_pixels_, inlier_bbox_min_, inlier_bbox_max_);
            len_lineseg_ = calcLineseg(inlier_bbox_min_, inlier_bbox_max_);

            // 点群密度が閾値未満の場合は無効化（num_inlier＝0）
            num_inlier_ = densityFilter(inlier_dense_th_);
        }
        else
        {
            num_inlier_ = 0;
        }

        if(num_inlier_ == 0)
        {
            inlier_pixels_.clear();
        }
    }

    return num_inlier_;
}

void FigLine::calcInlierBBox(const CvPointList &pixels, cv::Point &bbox_min, cv::Point &bbox_max) const
{
    bbox_min.x = INT_MAX;
    bbox_min.y = INT_MAX;
    bbox_max.x = INT_MIN;
    bbox_max.y = INT_MIN;

    for(const auto &px : pixels)
    {
        if(px.x < bbox_min.x)
        {
            bbox_min.x = px.x;
        }
        if(px.y < bbox_min.y)
        {
            bbox_min.y = px.y;
        }
        if(bbox_max.x < px.x)
        {
            bbox_max.x = px.x;
        }
        if(bbox_max.y < px.y)
        {
            bbox_max.y = px.y;
        }
    }
    return;
}

int FigLine::calcLineseg(const cv::Point &bbox_min, const cv::Point &bbox_max) const
{
    int len_lineseg;
    int bbox_w, bbox_h;

    bbox_w = bbox_max.x - bbox_min.x;
    bbox_h = bbox_max.y - bbox_min.y;
    len_lineseg = (bbox_w > bbox_h) ? bbox_w : bbox_h;

    return len_lineseg;
}

int FigLine::densityFilter(double density_th)
{
    int min_inlier_th;
    min_inlier_th = (int)(density_th * (double)len_lineseg_);

    if(num_inlier_ < min_inlier_th)
    {
        num_inlier_ = 0;
    }

    return num_inlier_;
}

void FigLine::calcIntersectBBox(const cv::Point &bbox_min, const cv::Point &bbox_max, CvPointList &inter_px) const
{
    double bmin_x, bmin_y, bmax_x, bmax_y;
    double x, y;
    
    bmin_x = (double)bbox_min.x;
    bmin_y = (double)bbox_min.y;
    bmax_x = (double)bbox_max.x;
    bmax_y = (double)bbox_max.y;

    inter_px.clear();

    // 上端(y=bmin_y)/下端(y=bmax_y)との交点
    if(fabs(a_) > 1e-5)
    {
        y = bmin_y;
        x = -(b_ * y + c_) / a_;
        if(((bmin_x - 1e-5) < x) && (x < (bmax_x + 1e-5)))
        {
            inter_px.push_back(cv::Point((int)x, (int)y));
        }

        y = bmax_y;
        x = -(b_ * y + c_) / a_;
        if(((bmin_x - 1e-5) < x) && (x < (bmax_x + 1e-5)))
        {
            inter_px.push_back(cv::Point((int)x, (int)y));
        }
    }

    // 左端(x=bmin_x)/右端(x=bmax_x)との交点
    if(fabs(b_) > 1e-5)
    {
        x = bmin_x;
        y = -(a_ * x + c_) / b_;
        if(((bmin_y - 1e-5) < y) && (y < (bmax_y + 1e-5)))
        {
            inter_px.push_back(cv::Point((int)x, (int)y));
        }

        x = bmax_x;
        y = -(a_ * x + c_) / b_;
        if(((bmin_y - 1e-5) < y) && (y < (bmax_y + 1e-5)))
        {
            inter_px.push_back(cv::Point((int)x, (int)y));
        }
    }

    return;
}

void FigLine::draw(cv::Mat &img) const
{
    const cv::Scalar COL = cv::Scalar(0,255,255);
    const double ALPHA = 0.6;

    CvPointList inter_px;

    // 直線と点群の外接矩形の交点を算出
    calcIntersectBBox(inlier_bbox_min_, inlier_bbox_max_, inter_px);

    if(inter_px.size() >= 2)
    {
        // 直線描画
        cv::Mat img_draw_layer;
        
        img_draw_layer = img.clone();

        cv::line(img_draw_layer, 
                 inter_px[0], inter_px[1],
                 COL, 2, cv::LINE_AA);

        cv::addWeighted(img_draw_layer, ALPHA, img, 1.0-ALPHA, 0.0, img);
    }

    return;
}

std::string FigLine::toString(void) const
{
    std::stringstream ss;
    ss << Fig::toString() << ",a=" << a_ << ",b=" << b_ << ",c=" << c_;
    ss << ",inlier_bbox={(" << inlier_bbox_min_.x << "," << inlier_bbox_min_.y << ")-";
    ss << "(" << inlier_bbox_max_.x << "," << inlier_bbox_max_.y << ")},";
    ss << "len_lineseg=" << len_lineseg_ << "}";
    return ss.str();
}

// === FigCircle ===

FigCircle& FigCircle::operator=(const FigCircle& f)
{
    Fig::operator=(f);

    a_ = f.a_;
    b_ = f.b_;
    c_ = f.c_;
    center_ = f.center_;
    r_      = f.r_;
    inlier_dense_th_ = f.inlier_dense_th_;
    min_r_th_        = f.min_r_th_;

    return (*this);
}

void FigCircle::operator=(const std::shared_ptr<Fig> &p)
{
    const std::shared_ptr<FigCircle> pcircle = std::dynamic_pointer_cast<FigCircle>(p);

    if(pcircle != nullptr)
    {
        FigCircle::operator=(*pcircle);
    }
    return;
}

void FigCircle::choiseRandomPixels(const CvPointList &pixels, CvPointList &sel_pixels) const
{
    size_t num_pixels;

    num_pixels = pixels.size();

    sel_pixels.clear();

    if(num_pixels >= 3)
    {
        // pixelsの中からランダムに3点を選ぶ（重複禁止）
        std::random_device rd;
        std::mt19937 rand_gen;
        std::uniform_int_distribution<size_t> dist(0, num_pixels-1);
        size_t idx1, idx2, idx3;
        cv::Point px1, px2, px3;
        
        rand_gen = std::mt19937(rd());

        idx1 = dist(rand_gen);
        do {
            idx2 = dist(rand_gen);
            idx3 = dist(rand_gen);
        } while ((idx1 == idx2) || (idx1 == idx3) || (idx2 == idx3)); // 同じ点が選ばれたらやり直し

        sel_pixels.push_back(pixels[idx1]);
        sel_pixels.push_back(pixels[idx2]);
        sel_pixels.push_back(pixels[idx3]);
    }

    return;
}

bool FigCircle::isEnableCreate(const CvPointList &sel_pixels) const
{
    bool is_enable;

    is_enable = false;

    if(sel_pixels.size() >= 3)
    {
        // 3点が一直線上にあるかどうかを判定
        //   → 3点で形成される三角形の面積が0かどうかで判定
        //   → 2ベクトルの外積が0かどうかで判定
        int x0,y0;
        int x1,y1;
        int x2,y2;
        int cross;

        x0 = sel_pixels[0].x; 
        y0 = sel_pixels[0].y;
        x1 = sel_pixels[1].x; 
        y1 = sel_pixels[1].y;
        x2 = sel_pixels[2].x; 
        y2 = sel_pixels[2].y;
        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        
        if(abs(cross) != 0)
        {
            // [入力3点が一直線上に存在しない] 円作成可
            is_enable = true;
        }
    }
    return is_enable;
}

void FigCircle::create(const CvPointList &sel_pixels) 
{
    if(sel_pixels.size() >= 3)
    {
        double x0,y0;
        double x1,y1;
        double x2,y2;
        double detA;
        double b0,b1,b2;
        double a,b,c;
        double cx,cy,r;

        x0 = (double)sel_pixels[0].x;
        y0 = (double)sel_pixels[0].y;
        x1 = (double)sel_pixels[1].x;
        y1 = (double)sel_pixels[1].y;
        x2 = (double)sel_pixels[2].x;
        y2 = (double)sel_pixels[2].y;

        // 円のパラメータa,b,cを算出(x^2 + y^2 + ax + by + c = 0)
        //   → 連立方程式AP=Bを解く。P=[a,b,c]

        // 行列式の計算 (サラスの方法)
        detA = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1);

        if(std::abs(detA) > 1e-5)
        {
            b0 = -(x0 * x0 + y0 * y0);
            b1 = -(x1 * x1 + y1 * y1);
            b2 = -(x2 * x2 + y2 * y2);

            // クラメルの公式で a, b, c を算出
            a = (b0 * (y1 - y2) - y0 * (b1 - b2) + (b1 * y2 - b2 * y1)) / detA;
            b = (x0 * (b1 - b2) - b0 * (x1 - x2) + (x1 * b2 - x2 * b1)) / detA;
            c = (x0 * (y1 * b2 - y2 * b1) - y0 * (x1 * b2 - x2 * b1) + b0 * (x1 * y2 - x2 * y1)) / detA;

            // a,b,cから中心(cx,cy), 半径rを算出
            cx = -a / 2.0;
            cy = -b / 2.0;
            r = std::sqrt(cx * cx + cy * cy - c);

            if(min_r_th_ < r)
            {
                a_ = a;
                b_ = b;
                c_ = c;
                center_.x = (int)cx;
                center_.y = (int)cy;
                r_ = (int)r;

                is_valid_ = true;
            }
        }
    }

    return;
}

int FigCircle::countInlier(const CvPointList &pixels, double dist_th)
{
    num_inlier_ = 0;
    inlier_pixels_.clear();

    if(is_valid_ == true)
    {
        // 点と直線の距離 < 閾値 を満たす点の数をカウント
        //    点と円周の距離＝|点と円中心の距離 - 円半径|

        //    平方根計算を回避するため、判定式を以下にする
        //      (円半径 - 閾値)^2 < 点と円中心の距離^2 < (円半径 + 閾値)^2
        cv::Point vec_px_center;
        double dist2;
        double r_min2, r_max2;

        r_min2  = r_ - dist_th;
        r_min2 *= r_min2;

        r_max2  = r_ + dist_th;
        r_max2 *= r_max2;

        for(const auto &px : pixels)
        {
            vec_px_center.x = px.x - center_.x;
            vec_px_center.y = px.y - center_.y;
            dist2 = (double)(vec_px_center.x * vec_px_center.x + vec_px_center.y * vec_px_center.y);

            if((r_min2 < dist2) && (dist2 < r_max2))
            {
                num_inlier_++;
                inlier_pixels_.push_back(px);
            }
        }

        if(num_inlier_ > min_inlier_th_)
        {
            // 点群密度が閾値未満の場合は無効化（num_inlier＝0）
            num_inlier_ = densityFilter(inlier_dense_th_);
        }
        else
        {
            num_inlier_ = 0;
        }

        if(num_inlier_ == 0)
        {
            inlier_pixels_.clear();
        }
    }

    return num_inlier_;
}

int FigCircle::densityFilter(double density_th)
{
    double len_circle;
    double min_inlier_th;

    len_circle = 2.0 * M_PI * (double)r_;

    min_inlier_th = (int)(len_circle * density_th);

    if(num_inlier_ < min_inlier_th)
    {
        num_inlier_ = 0;
    }

    return num_inlier_;
}

void FigCircle::draw(cv::Mat &img) const
{
    const cv::Scalar COL = cv::Scalar(0,255,255);
    const double ALPHA = 0.6;

    cv::Mat img_draw_layer;

    // 円描画
    img_draw_layer = img.clone();

    cv::circle(img_draw_layer, center_, r_, COL, 2, cv::LINE_AA);

    cv::addWeighted(img_draw_layer, ALPHA, img, 1.0-ALPHA, 0.0, img);

    return;
}

std::string FigCircle::toString(void) const
{
    std::stringstream ss;
    ss << Fig::toString() << ",a=" << a_ << ",b=" << b_ << ",c=" << c_;
    ss << ",center=(" << center_.x << "," << center_.y << ")";
    ss << ",r=" << r_ << "}";
    return ss.str();
}
