#ifndef _FIG_HPP_
#define _FIG_HPP_

#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "main.hpp"

typedef std::vector<cv::Point> CvPointList;

typedef struct 
{
    cv::Point pt_min_;
    cv::Point pt_max_;

    void clear(void)
    {
        pt_min_.x = pt_min_.y = 0;
        pt_max_.x = pt_max_.y = 0;
        return;
    }
} CvRect;

class Vec2 
{
public:
    double x_, y_;

    Vec2()
     : x_(0.0), y_(0.0) 
    {
        return;
    }
    Vec2(double x, double y)
     : x_(x), y_(y) 
    {
        return;
    }
    Vec2(const cv::Point &p)
     : x_((double)p.x), y_((double)p.y)
    {
        return;
    }

    Vec2 operator+(const Vec2& o) const 
    {
        return Vec2(x_ + o.x_, y_ + o.y_); 
    }
    Vec2 operator-(const Vec2& o) const 
    {
        return Vec2(x_ - o.x_, y_ - o.y_); 
    }
    Vec2 operator*(double s) const 
    {
        return Vec2(x_ * s, y_ * s); 
    }
    double dot(const Vec2& a) const
    {
        return x_ * a.x_ + y_ * a.y_;
    }
    Vec2 normalize(void) const
    {
        double n = sqrt(x_ * x_ + y_ * y_);
        return Vec2(x_ / n, y_ / n);
    }
};

class FigType
{
public:
    enum 
    {
        FIGTYPE_CIRCLE_,
        FIGTYPE_LINE_,
        FIGTYPE_NONE_
    };

    FigType()
    {
        figtype_ = FIGTYPE_CIRCLE_;
        return;
    }

    void next(void)
    {
        if(figtype_ == FIGTYPE_CIRCLE_)
        {
            figtype_ = FIGTYPE_LINE_;
        }
        else if(figtype_ == FIGTYPE_LINE_)
        {
            figtype_ = FIGTYPE_NONE_;
        }
        else
        {
        }
        return;
    }
    
    bool isNone() const
    {
        return figtype_ == FIGTYPE_NONE_;
    }
    
    std::string toString(void) const
    {
        std::string ret_str("NONE");
        if(figtype_ == FIGTYPE_CIRCLE_)
        {
            ret_str = std::string("CIRCLE");
        }
        else if(figtype_ == FIGTYPE_LINE_)
        {
            ret_str = std::string("LINE");
        }
        else
        {
        }
        return ret_str;
    }

    int figtype_;

};

class Fig
{
public:
    Fig(CfgType &cfg) 
    {
        is_valid_ = false;
        num_inlier_ = 0;
        inlier_pixels_.clear();
        inlier_bbox_.pt_min_ = cv::Point(0,0);
        inlier_bbox_.pt_max_ = cv::Point(0,0);

        inlier_dense_th_ = 0.0;
        dist_th_ = strtof(cfg["INLIER_DIST_TH"].c_str(), NULL);
        min_inlier_th_ = strtol(cfg["INLIER_NUM_MIN_TH"].c_str(), NULL, 10);
        return;
    }
    virtual ~Fig() 
    {
        return;
    }

    void reset(void)
    {
        is_valid_ = false;
        num_inlier_ = 0;
        inlier_pixels_.clear();
        inlier_bbox_.clear();
        return;
    }
    Fig& operator=(const Fig& f);
    virtual void operator=(const std::shared_ptr<Fig> &p);

    virtual std::shared_ptr<Fig> clone() const = 0;
    
    virtual void choiseRandomPixels(const CvPointList &pixels, CvPointList &sel_pixels) const
    {
        return;
    }
    virtual bool isEnableCreate(const CvPointList &sel_pixels) const 
    {
        return false;
    }
    virtual void create(const CvPointList &sel_pixels) 
    {
        return;
    }
    virtual int countInlier(const CvPointList &pixels, double dist_th)
    {
        return 0;
    }

    void calcInlierBBox(void);

    virtual bool filteredByInlierPixels(void)
    {
        return false;
    }

    virtual void erasePixels(cv::Mat &img) const;

    virtual void draw(cv::Mat &img) const
    {
        return;
    }
    virtual std::string toString(void) const;

    bool is_valid_;
    int  num_inlier_;
    CvPointList inlier_pixels_;
    CvRect inlier_bbox_;

    double inlier_dense_th_;
    double dist_th_;
    int min_inlier_th_;

};

class FigLine : public Fig
{
public:
    FigLine(CfgType &cfg) : Fig(cfg)
    {
        a_ = 0.0;
        b_ = 0.0;
        c_ = 0.0;
        sqrt_a2_plus_b2_ = 0.0;

        len_lineseg_ = 0;
        lineseg_pt0_ = cv::Point(0,0);
        lineseg_pt1_ = cv::Point(0,0);

        inlier_dense_th_ = strtof(cfg["INLIER_LINE_DENSE_TH"].c_str(), NULL);
        line_min_len_th_ = strtol(cfg["LINE_MIN_LEN_TH"].c_str(), NULL, 10);
        return;
    }

    FigLine& operator=(const FigLine& f);
    void operator=(const std::shared_ptr<Fig> &p) override;

    std::shared_ptr<Fig> clone() const override 
    {
        return std::make_shared<FigLine>(*this);
    }

    void choiseRandomPixels(const CvPointList &pixels, CvPointList &sel_pixels) const override;
    bool isEnableCreate(const CvPointList &sel_pixels) const override;
    void create(const CvPointList &sel_pixels) override;
    int countInlier(const CvPointList &pixels, double dist_th) override;
    int calcLenLineseg(void) const;
    bool densityFilter(double density_th);
    void extractLineSegPixels(double k);
    bool filteredByInlierPixels(void) override;
    void calcIntersectBBox(const cv::Point &bbox_min, const cv::Point &bbox_max, CvPointList &inter_px) const;
    void draw(cv::Mat &img) const override;
    std::string toString(void) const override;

    double a_;
    double b_;
    double c_;
    double sqrt_a2_plus_b2_;

    int len_lineseg_;
    cv::Point lineseg_pt0_;
    cv::Point lineseg_pt1_;

    int line_min_len_th_;
};

class FigCircle : public Fig
{
public:
    FigCircle(CfgType &cfg) : Fig(cfg)
    {
        // x^2 + y^2 + ax + by + c = 0
        a_ = 0.0;
        b_ = 0.0;
        c_ = 0.0;
        
        // 中心center、半径r
        center_ = cv::Point(0,0);
        r_ = 0;

        inlier_dense_th_ = strtof(cfg["INLIER_CIRCLE_DENSE_TH"].c_str(), NULL);
        min_r_th_ = strtof(cfg["CIRCLE_MIN_R_TH"].c_str(), NULL);
        return;
    }

    FigCircle& operator=(const FigCircle& f);
    void operator=(const std::shared_ptr<Fig> &p) override;

    std::shared_ptr<Fig> clone() const override 
    {
        return std::make_shared<FigCircle>(*this);
    }

    void choiseRandomPixels(const CvPointList &pixels, CvPointList &sel_pixels) const override;
    bool isEnableCreate(const CvPointList &sel_pixels) const override;
    void create(const CvPointList &sel_pixels) override;
    int countInlier(const CvPointList &pixels, double dist_th) override;
    bool densityFilter(double density_th);
    bool filteredByInlierPixels(void) override;
    void erasePixels(cv::Mat &img) const override;
    void draw(cv::Mat &img) const override;
    std::string toString(void) const override;

    double a_;
    double b_;
    double c_;

    cv::Point center_;
    int r_;

    double inlier_dense_th_;
    double min_r_th_;
    
};

typedef std::vector<std::shared_ptr<Fig>> FigList;

#endif // _FIG_HPP_
