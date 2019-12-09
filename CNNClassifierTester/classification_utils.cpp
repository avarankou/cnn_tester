#include "classification_utils.h"

#include <filetree_rambler.h>
#include <file_utils.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef WITH_OPENCV_HIGHGUI
#include <opencv2/highgui.hpp>
#include <iomanip>
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
        params.annotation = cmd.get<int>("annotation");
        if (params.annotation == 2)
            params.thumbnails_dir = (path)cmd.get<std::string>("thumbnails_dir");
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
            const int SPACE_KEY = 32;
            const int ENTER_KEY = 13;
            const std::string win_label("classification");
            const std::string recheck_win_lbl("RECHECK classification");

            bool new_outname = params.outname_from_classification;
            bool mis = false;

            cv::Mat img = cv::imread(file.string(), cv::IMREAD_COLOR);
            cv::Mat dbg_img;
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

            if (params.recheck_misclassified || params.dbg || params.annotation)
            {
                if (params.annotation <= 1)
                    label_img(img, dbg_img, result);
                else
                    label_img_with_thumbnails(img, dbg_img, result);
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

    void label_img(const cv::Mat & img, cv::Mat & labeled, const clf::classifier_t::clf_res_t & result)
    {
        const cv::Scalar COLOR_RED(0, 0, 255);
        const cv::Scalar COLOR_GREEN(0, 255, 0);
        const cv::Scalar COLOR_BLUE(255, 0);

        labeled = img;

        if (result.rec.empty())
            return;

        int label_rows = (classifier.params.check_filename ? 1 : 0);
        size_t max_pred_topk = 0;
        std::stringstream gt;
        for (size_t i = 0; i < classifier.params.num_classes; ++i)
        {
            max_pred_topk = std::max(max_pred_topk, result.rec[i].size());
            if (result.gt.size() <= i)
                continue;
            gt << result.gt[i] << ' ';
        }
        label_rows += (int)max_pred_topk;

        if (label_rows == 0)
           return;

        int row_height = 20;
        int print_pos = img.rows + row_height / 2;

        cv::Scalar text_bg = cv::Scalar(255, 255, 255);
        labeled = cv::Mat(img.rows + label_rows * row_height, img.cols, CV_8UC3, text_bg);
        img.copyTo(cv::Mat(labeled, cv::Rect(0, 0, img.cols, img.rows)));

        if (classifier.params.check_filename)
        {
            cv::putText(labeled, gt.str(), cv::Point(10, print_pos), 1, 1.5, COLOR_BLUE);
            print_pos += row_height;
        }

        for (size_t i = 0; i < max_pred_topk; ++i, print_pos += row_height)
        {
            std::stringstream rec;
            for (size_t j = 0; j < classifier.params.num_classes; ++j)
            {
                if (result.rec[j].size() <= i)
                    continue;
                std::string class_name = result.rec[j][i].first;
                float rel = result.rec[j][i].second;
                rec << class_name << ' ' << std::setprecision(2) << rel;
            }

            if (i == 0 && result.correct)
                cv::putText(labeled, rec.str(), cv::Point(10, print_pos), 1, 1., COLOR_GREEN);
            else
                cv::putText(labeled, rec.str(), cv::Point(10, print_pos), 1, 1., COLOR_RED);
        }
    }

    void label_img_with_thumbnails(const cv::Mat & img, cv::Mat & labeled, const clf::classifier_t::clf_res_t & result)
    {
        const cv::Scalar COLOR_RED(0, 0, 255);
        const cv::Scalar COLOR_GREEN(0, 255, 0);
        const cv::Scalar COLOR_BLUE(255, 0);

        labeled = img;

        // only for single-label classification
        if (result.rec.empty() || result.rec[0].empty())
            return;

        int label_rows = (classifier.params.check_filename ? 1 : 0);
        size_t pred_topk = result.rec[0].size();
        label_rows += (int)pred_topk;

        if (label_rows == 0)
            return;

        int thumb_height = 128;
        int thumb_width = 128;

        cv::Mat img_rs = img;
        if (img.cols < thumb_width)
        {
            double sf = 1. * thumb_width / img.cols;
            cv::resize(img, img_rs, cv::Size(), sf, sf);
        }

        int print_pos = img_rs.rows + thumb_height / 2;
        cv::Scalar text_bg = cv::Scalar(255, 255, 255);
        int labeled_height = img_rs.rows + label_rows * thumb_height;
        int labeled_width = (img_rs.cols + thumb_width);
        labeled = cv::Mat(labeled_height, labeled_width, CV_8UC3, text_bg);
        img_rs.copyTo(cv::Mat(labeled, cv::Rect(thumb_width, 0, img_rs.cols, img_rs.rows)));

        if (classifier.params.check_filename && !result.gt.empty())
        {
            cv::putText(labeled, result.gt[0], cv::Point(thumb_width + 10, print_pos), 1, 1.5, COLOR_BLUE);
            print_pos += thumb_height;
        }

        for (size_t i = 0; i < pred_topk; ++i, print_pos += thumb_height)
        {
            std::stringstream rec;
            if (result.rec[0].size() <= i)
                break;
            std::string class_name = result.rec[0][i].first;
            float rel = result.rec[0][i].second;
            rec << class_name << ' ' << std::setprecision(2) << rel;

            cv::Mat thumbnail = cv::imread((params.thumbnails_dir / (class_name + ".jpg")).string(), cv::IMREAD_COLOR);
            if (!thumbnail.empty())
            {
                cv::resize(thumbnail, thumbnail, cv::Size(thumb_width, thumb_height));
                cv::Mat roi(labeled, cv::Rect(0, print_pos - (thumb_height / 2), thumb_width, thumb_height));
                thumbnail.copyTo(roi);
            }

            if (i == 0 && result.correct)
                cv::putText(labeled, rec.str(), cv::Point(thumb_width + 10, print_pos), 1, 1., COLOR_GREEN);
            else
                cv::putText(labeled, rec.str(), cv::Point(thumb_width + 10, print_pos), 1, 1., COLOR_RED);
        }
    }
}