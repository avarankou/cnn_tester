#pragma once

#include "classification.h"

#include <filesystem>

#define WITH_OPENCV_HIGHGUI

namespace ct
{
    using path = clf::path;


    struct param_t
    {
        path indir;
        path outdir;
        path misdir;
        path thumbnails_dir;
        int save_misclassified;
        int indir_mode;
        int outdir_mode;
        bool outname_from_classification;
        int annotation;
        bool move_out;
#       ifdef WITH_OPENCV_HIGHGUI
        bool dbg;
        bool recheck_misclassified;
#       endif //WITH_OPENCV_HIGHGUI
    };

    bool init_params(const cv::CommandLineParser & cmd);
    void process_dir();

    /* thread func */
    void process_file(const path & file, std::mutex & dnn_mutex);
    void label_img(const cv::Mat & img, cv::Mat & labeled, const clf::classifier_t::clf_res_t & result);
    void label_img_with_thumbnails(const cv::Mat & img, cv::Mat & labeled, const clf::classifier_t::clf_res_t & result);
}