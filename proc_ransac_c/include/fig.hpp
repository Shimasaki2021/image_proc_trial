#ifndef _FIG_HPP_
#define _FIG_HPP_

#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "main.hpp"

typedef std::vector<cv::Point> CvPointList;

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
        return;
    }

    void erasePixels(cv::Mat &img) const;
    
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

    bool is_valid_;
    int  num_inlier_;
    CvPointList inlier_pixels_;

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
        inlier_dense_th_ = strtof(cfg["INLIER_LINE_DENSE_TH"].c_str(), NULL);
        return;
    }

    void choiseRandomPixels(const CvPointList &pixels, CvPointList &sel_pixels) const;
    bool isEnableCreate(const CvPointList &sel_pixels) const;
    void create(const CvPointList &sel_pixels);
    int countInlier(const CvPointList &pixels, double dist_th);
    void calcInlierBBox(const CvPointList &pixels, cv::Point &bbox_min, cv::Point &bbox_max) const;
    int calcLineseg(const cv::Point &bbox_min, const cv::Point &bbox_max) const;
    int densityFilter(double density_th);

    double a_;
    double b_;
    double c_;
    double sqrt_a2_plus_b2_;

    cv::Point inlier_bbox_min_;
    cv::Point inlier_bbox_max_;
    int len_lineseg_;


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
        min_r_th_ = strtol(cfg["CIRCLE_MIN_R_TH"].c_str(), NULL, 10);
        return;
    }

    void choiseRandomPixels(const CvPointList &pixels, CvPointList &sel_pixels) const;
    bool isEnableCreate(const CvPointList &sel_pixels) const;
    void create(const CvPointList &sel_pixels);
    int countInlier(const CvPointList &pixels, double dist_th);
    int densityFilter(double density_th);

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
