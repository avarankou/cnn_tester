#pragma once

#include "detection.h"

#include <filesystem>

#define WITH_OPENCV_HIGHGUI

namespace dt
{
    using path = detector::path;

    struct param_t
    {
        path indir;
        int indir_mode;
        path outdir;
        int outdir_mode;
        bool move_out;
        double min_width;
        double min_height;
        size_t max_objects;
#       ifdef WITH_OPENCV_HIGHGUI
        bool dbg;
        bool recheck_falses;
#       endif // WITH_OPENCV_HIGHGUI
    };

    bool init_params(const cv::CommandLineParser & cmd);
    void process_dir();

    /* thread func */
    static void process_file(const path & file, std::mutex & dnn_mutex);
}