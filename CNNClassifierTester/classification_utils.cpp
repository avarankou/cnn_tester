#include "classification_utils.h"

#include <filetree_rambler.h>
#include <file_utils.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef WITH_OPENCV_HIGHGUI
#include <opencv2/highgui.hpp>
#endif // WITH_OPENCV_HIGHGUI

namespace ct
{
    using LL = logger::LOG_LEVEL_t;

    static clf::classifier_t classifier;
    static param_t params;

    bool init_params(const cv::CommandLineParser & cmd)
    {
        bool retval = true;

        if (!cmd.has("indir"))
        {
            logger::LOG_MSG(LL::Error, "<indir> must be specified.\n");
            retval = false;
        }
        params.indir = (path)cmd.get<std::string>("indir");

        int indir_mode = cmd.get<int>("indir_mode");
        params.indir_mode = indir_mode;
        if (indir_mode > 1)
        {
            logger::LOG_MSG(LL::Error, "Wrong value <indir_mode>=" + std::to_string(indir_mode) + ". 0 - root folder only, n - allowed subdirs depth (-1 for any depth).");
            retval = false;
        }

        int outdir_mode = cmd.get<int>("outdir_mode");
        if (outdir_mode < -1 || outdir_mode > 1)
        {
            logger::LOG_MSG(LL::Error, "Wrong value <outdir_mode>=" + std::to_string(outdir_mode) + ". Allowed values: -1 - disable output, 0 - common root folder, 1 - separate folders.");
            retval = false;
        }
        params.outdir_mode = outdir_mode;

        if (!cmd.has("outdir") && outdir_mode >= 0)
        {
            logger::LOG_MSG(LL::Error, "<outdir> must be specified\n");
            retval = false;
        }
        params.outdir = (path)cmd.get<std::string>("outdir");
        params.outname_from_classification = cmd.get<int>("out_filename") == 0 ? true : false;
        params.annotation = cmd.get<int>("annotation") == 0 ? false : true;
        params.move_out= cmd.get<int>("move_out") == 0 ? false : true;


        params.misdir = (path)cmd.get<std::string>("misclassified_dir");
        int save_mis = cmd.get<int>("save_misclassified");
        if (save_mis < -1 || save_mis > 1)
        {
            logger::LOG_MSG(LL::Error, "Wrong value <save_misclassified>=" + std::to_string(save_mis) + ". Allowed values: -1 - don't, 0 - filename from classified labels, 1 - original filename.");
            retval = false;
        }
        params.save_misclassified = save_mis;

#       ifdef WITH_OPENCV_HIGHGUI
            params.dbg = cmd.get<int>("debug_win") == 0 ? false : true;
            params.recheck_misclassified = cmd.get<int>("recheck_misclassified") == 0 ? false : true;
            if (!params.dbg && !params.recheck_misclassified&& params.outdir_mode < 0 && params.save_misclassified < 0)
                logger::LOG_MSG(LL::Warning, "Wrong values possible. <outdir_mode>=-1, <recheck_misclassified>=-1, <save_misclassified>=-1, and <debug_win>=-1 the program will run without any output and effect.");
#       endif // WITH_OPENCV_HIGHGUI

        if (retval)
            retval = classifier.init_params(cmd) && classifier.load_classifier();

        if (params.recheck_misclassified && !classifier.params.check_filename)
        {
            logger::LOG_MSG(LL::Warning, "Wrong values possible. <recheck_misclassified>=1 and <filename_as_labels>=0 classification results can't be compared to ground truth.");
            params.recheck_misclassified = false;
        }

        return retval;
    }

    void process_dir()
    {
        if (params.outdir_mode != -1)
            ftr::create_dir(params.outdir);
        if (params.save_misclassified != -1)
            ftr::create_dir(params.misdir);

#       ifdef WITH_OPENCV_HIGHGUI
            if (params.recheck_misclassified)
                logger::LOG_MSG(LL::Info,
                    "Recheck_misclassified = 1"
                    " If image classification result doesn't match ground truth from it's filename"
                    " it is shown for manual recheck."
                    "Press SPACE to save image in <outdir> with original filename,"
                    "ENTER for filename from classification result,"
                    "any key to save image to <misdir>");
#       endif // WITH_OPENCV_HIGHGUI

        ftr::settings_t settings;
        settings.check_subdirs = (params.indir_mode != 0);
        settings.max_subdir_depth = (params.indir_mode == -1 ? 0u : (size_t)params.indir_mode);
        settings.ext_list = { ".jpg", ".jpeg", ".png", ".bmp" };
#       ifdef WITH_OPENCV_HIGHGUI
            if (params.dbg || params.recheck_misclassified)
                settings.max_threads = 1u; // TODO: check cv::waitkey() threadsafety
            else
                settings.max_threads = 16u;
#       else
            settings.max_threads = 16u;
#       endif // WITH_OPENCV_HIGHGUI

        ftr::scan(params.indir, process_file, settings);
        classifier.print_stat();
    }

    void process_file(const path & file, std::mutex & dnn_mutex)
    {
        try
        {
            const cv::Scalar COLOR_RED(0, 0, 255);
            const cv::Scalar COLOR_GREEN(0, 255, 0);
            const int SPACE_KEY = 32;
            const int ENTER_KEY = 13;
            const std::string win_label("classification");
            const std::string recheck_win_lbl("RECHECK classification");

            bool new_outname = params.outname_from_classification;
            bool mis = false;

            cv::Mat img = cv::imread(file.string());
            if (img.empty())
            {
                logger::LOG_MSG(LL::Warning, "Failed to load image: " + file.string());
                return;
            }

            clf::classifier_t::clf_res_t result;
            {
                std::lock_guard<std::mutex> lg(dnn_mutex);
                classifier.process_file(file, img, result);
            }

            cv::Mat dbg_img;
            if (params.recheck_misclassified || params.dbg || params.annotation)
            {
                img.copyTo(dbg_img);

                std::stringstream gt;
                std::stringstream rec;
                for (size_t i = 0; i < classifier.params.num_classes; ++i)
                {
                    gt << result.gt[i] << ' ';
                    rec << result.rec[i] << ' ';
                }
                if (classifier.params.check_filename)
                {
                    cv::putText(dbg_img, gt.str(), cv::Point(10, 10), 1, 1, COLOR_GREEN);
                    if (!result.correct)
                        cv::putText(dbg_img, rec.str(), cv::Point(10, 30), 1, 1, COLOR_RED);
                }
                else
                    cv::putText(dbg_img, rec.str(), cv::Point(10, 30), 1, 1, COLOR_RED);

            }
#           ifdef WITH_OPENCV_HIGHGUI
            if (params.recheck_misclassified &&
                !result.correct)
            {
                cv::imshow(recheck_win_lbl, dbg_img);

                int key = cv::waitKey(0) % 256;
                if (key == SPACE_KEY)
                    new_outname = false;
                else if (key == ENTER_KEY)
                    new_outname = true;
                else
                {
                    mis = true;
                    new_outname = params.save_misclassified == 0;
                }
            }
            else if (params.dbg)
            {
                cv::imshow(win_label, dbg_img);
                cv::waitKey(0);
            }
            else if (params.save_misclassified >= 0)
                mis = !result.correct;
#           else
                mis = (params.save_misclassified > 0 && !result.correct);
#           endif // WITH_OPENCV_HIGHGUI

            /* disable output */
            if (params.outdir_mode < 0 && params.save_misclassified < 0)
                return;

            path dst;
            if (mis)
                dst = params.misdir;
            else if (params.outdir_mode >= 0)
                dst = params.outdir;
            else
                return;

            if (params.outdir_mode == 1)
            {
                dst.append(ftr::subdirs(file.parent_path(), params.indir));
                ftr::create_dir(dst);
            }
            dst.append(new_outname ?
                classifier.change_filename(file.filename().string(), result.rec) :
                file.filename());

            params.annotation ?
                cv::imwrite(dst.string(), dbg_img) :
                cv::imwrite(dst.string(), img);

            if (params.move_out)
                ftr::remove_file(file);
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