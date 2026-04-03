#ifndef _DEBUG_HPP_
#define _DEBUG_HPP_

#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>

class DebugOut
{
public:
    DebugOut(const char *outdir, const char *fname_base);
    ~DebugOut();

    void createOutdir(void) const;
    void openLogFile(const char *fname);
    void closeLogFile(void);

    void dumpImg(const cv::Mat &img, const std::string &postfix) const;
    int printLogLine(const char *fmt,...);

    std::string outdir_;
    std::string fname_base_;
    std::ofstream log_fp_;
    bool is_out_;

};

#endif // _DEBUG_HPP_
