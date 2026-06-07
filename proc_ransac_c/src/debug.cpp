#include <sys/types.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <fstream>
#include "debug.hpp"

DebugOut::DebugOut(const char *outdir, const char *fname_base)
    : outdir_(outdir), fname_base_(fname_base), is_out_(false)
{
    return;
}

DebugOut::~DebugOut()
{
    closeLogFile();
    return;
}

void DebugOut::createOutdir(void) const
{
    if(is_out_ == true)
    {
        struct stat s;
        int ret;
        char command[128];

        ret = stat(outdir_.c_str(), &s);
        if((ret == 0) && (s.st_mode & __S_IFDIR))
        {
            sprintf(command,"rm -rf %s",outdir_.c_str());
            system(command);
        }
        sprintf(command,"mkdir %s",outdir_.c_str());
        system(command);
    }
    return;
}

void DebugOut::openLogFile(const char *fname)
{
    closeLogFile();
    if(is_out_ == true)
    {
        std::string fpath;
        createOutdir();

        fpath = outdir_ + "/" + std::string(fname);
        log_fp_.open(fpath, std::ios::out);
    }

    return;
}

void DebugOut::closeLogFile(void)
{
    if(log_fp_.is_open() == true)
    {
        log_fp_.close();
    }
    return;
}

void DebugOut::dumpImg(const cv::Mat &img, const std::string &postfix) const
{
    if(is_out_ == true)
    {
        std::string fpath;

        fpath = outdir_ + "/" + fname_base_ + postfix + ".png";
        cv::imwrite(fpath, img);
    }
    return;
}

int DebugOut::printLogLine(const char *fmt,...)
{
    int ret;

    ret = 0;

    if(log_fp_.is_open() == true)
    {
        char buffer[256];
        va_list args;

        va_start(args, fmt);
        ret = vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args); 

        log_fp_ << buffer << std::endl;
    }

    return ret;
}
