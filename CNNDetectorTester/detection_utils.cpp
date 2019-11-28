#include "detection_utils.h"

#include <filetree_rambler.h>
#include <file_utils.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef WITH_OPENCV_HIGHGUI
#include <opencv2/highgui.hpp>
#endif // WITH_OPENCV_HIGHGUI

#include <iostream>
#include <fstream>

namespace dt
{
    using LL = logger::LOG_LEVEL_t;

    static detector::detector_t detector;
    static param_t params;

    bool init_params(const cv::CommandLineParser & cmd)
    {
        bool retval = true;

        if (!cmd.has("indir"))
        {
            logger::LOG_MSG(LL::Error, "<indir> must be specified.");
            retval = false;
        }
        params.indir = path(cmd.get<std::string>("indir"));

        int indir_mode = cmd.get<int>("indir_mode");
        params.indir_mode = indir_mode;
        if (indir_mode < -1 || indir_mode > 1)
        {
            logger::LOG_MSG(LL::Error, "Wrong value <indir_mode>=" + std::to_string(indir_mode) + ". Allowed values: 0 - root folder only, n - allowed subdirs depth (-1 for any depth).");
            retval = false;
        }

        int outdir_mode = cmd.get<int>("outdir_mode");
        if (outdir_mode < -1 || outdir_mode > 1)
        {
            logger::LOG_MSG(LL::Error, "Wrong value <outdir_mode>=" + std::to_string(indir_mode) + ". Allowed values: -1 - disable output, 0 - common root folder, 1 - separate folders.");
            retval = false;
        }
        params.outdir_mode = outdir_mode;

        if (outdir_mode != -1 && !cmd.has("outdir"))
        {
            logger::LOG_MSG(LL::Error, "<outdir> must be specified.");
            retval = false;
        }
        params.outdir = path(cmd.get<std::string>("outdir"));

        params.move_out = cmd.get<int>("move_out") == 0 ? false : true;

        params.min_height = cmd.get<double>("min_height");
        params.min_width = cmd.get<double>("min_width");
        params.max_objects = cmd.get<size_t>("max_objects");

#       ifdef WITH_OPENCV_HIGHGUI
            params.dbg = cmd.get<bool>("debug_win");
            params.recheck_falses = cmd.get<int>("recheck_falses") == 0 ? false : true;

            if (!params.dbg && !params.recheck_falses && params.outdir_mode < 0)
                logger::LOG_MSG(LL::Warning, "Wrong values possible. <outdir_mode>=-1 and <recheck_falses>=-1 and <debug_win>=-1 the program will run without any output and effect.");
#       endif // WITH_OPENCV_HIGHGUI

        if (retval)
            retval = detector.init_params(cmd) && detector.load_detector();

        return retval;
    }
    
    void process_dir()
    {
        if (params.outdir_mode != -1)
            ftr::create_dir(params.outdir);

#       ifdef WITH_OPENCV_HIGHGUI
            if (params.recheck_falses)
                logger::LOG_MSG(LL::Info,
                "Recheck_falses = 1"
                " Detected objects of lower size than desired or input images with no detections above specified threshold"
                " are shown for manual recheck."
                "Press SPACE to save image / crop to output, ENTER to skip");
#       endif // WITH_OPENCV_HIGHGUI

        ftr::settings_t settings;
        settings.check_subdirs = (params.indir_mode != 0);
        settings.max_subdir_depth = (params.indir_mode == -1 ? 0u : (size_t)params.indir_mode);
        settings.ext_list = { ".jpg", ".jpeg", ".png", ".bmp" };
#       ifdef WITH_OPENCV_HIGHGUI
            if (params.dbg || params.recheck_falses)
                settings.max_threads = 1u; // TODO: check cv::waitkey() threadsafety
            else
                settings.max_threads = 16u;
#       else
        settings.max_threads = 16u;
#       endif // WITH_OPENCV_HIGHGUI

        ftr::scan(params.indir, process_file, settings);
    }

    void move_file(const std::experimental::filesystem::v1::path & file, const cv::Mat & crop, int crop_idx = -1)
    {
        std::string & filename_short = file.filename().string();
        path dst = params.outdir;
        path subdir = ftr::subdirs(file.parent_path(), params.indir);

        if (!subdir.empty())
        {
            dst.append(subdir);
            ftr::create_dir(dst);
        }

        if (crop_idx != -1)
        {
            std::stringstream name;
            name << filename_short.substr(0, filename_short.find_last_of('.')) << '_' << crop_idx << file.extension();
            dst.append(name.str());
        }
        else
            dst.append(filename_short);

        cv::imwrite(dst.string(), crop);

        if (params.move_out)
            ftr::remove_file(file);
    }

    static void process_file(const path & file, std::mutex & dnn_mutex)
    {
        try
        {
            const cv::Scalar COLOR_RED(0, 0, 255);
            const cv::Scalar COLOR_GREEN(0, 255, 0);
            const int LINE_THICKNESS = 2;
            const int SPACE_KEY = 32;
            const int ENTER_KEY = 13;
            const std::string win_lbl("detection");
            const std::string recheck_win_lbl("RECHECK detection");

            cv::Mat img = cv::imread(file.string());
            if (img.empty())
            {
                logger::LOG_MSG(LL::Warning, "Failed to load image: " + file.string());
                return;
            }

            //if (img.channels() < 3 || (img.type() != CV_8UC3 && img.type() != CV_8UC4))
            //    return;

            std::vector<detector::detector_t::det_res_t > results;
            {
                std::lock_guard<std::mutex> lg(dnn_mutex);
                detector.process_file(img, results);
            }

            size_t num_res = std::min(results.size(), params.max_objects);
            if (num_res == 0)
                return;

            int crop_id = 0;

#       ifdef WITH_OPENCV_HIGHGUI
            cv::Mat dbg_img;
            if (params.dbg && num_res)
            {
                img.copyTo(dbg_img);
            }
#       endif // WITH_OPENCV_HIGHGUI

            for (size_t i = 0; i < num_res; ++i)
            {
                const auto & res = results[i];
                cv::Rect roi(res.x, res.y, res.w, res.h);
                bool crop = (res.w >= (int)(img.cols * params.min_width) &&
                    res.h >= (int)(img.rows * params.min_height));

#           ifdef WITH_OPENCV_HIGHGUI
                if (params.dbg)
                    cv::rectangle(dbg_img, roi, crop ? COLOR_GREEN : COLOR_RED, LINE_THICKNESS);

                if (params.recheck_falses && !crop)
                {
                    cv::Mat recheck_img;
                    img.copyTo(recheck_img);
                    cv::rectangle(recheck_img, roi, COLOR_RED, LINE_THICKNESS);

                    cv::imshow(recheck_win_lbl, recheck_img);
                    int key = cv::waitKey(0) % 256;
                    if (key == SPACE_KEY)
                        crop = true;
                }
#           endif // WITH_OPENCV_HIGHGUI

                if (params.outdir_mode < 0 || !crop)
                    continue;

                num_res == 1 ?
                    move_file(file, cv::Mat(img, roi)) :
                    move_file(file, cv::Mat(img, roi), crop_id++);
            }

#       ifdef WITH_OPENCV_HIGHGUI
            if (params.dbg && (num_res || !params.recheck_falses))
            {
                num_res ?
                    cv::imshow(win_lbl, dbg_img) :
                    cv::imshow(win_lbl, img);
                cv::waitKey(0);
            }
            else if (params.recheck_falses && !num_res)
            {
                cv::imshow(recheck_win_lbl, img);

                if (cv::waitKey(0) % 256 == SPACE_KEY)
                {
                    move_file(file, img);
                }
            }
#       endif // WITH_OPENCV_HIGHGUI
        }
        catch (cv::Exception & e)
        {
            logger::LOG_MSG(LL::Error, e.what());
        }
        catch (std::exception & e)
        {
            logger::LOG_MSG(LL::Error, e.what());
        }
        catch (...)
        {
            logger::LOG_MSG(LL::Error, "Unknown exception.");
        }
    }
}