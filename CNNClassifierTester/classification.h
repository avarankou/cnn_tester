#pragma once

#include "confusion_mat.h"

#include <opencv2/dnn.hpp>

#include <array>
#include <filesystem>

namespace clf
{
    using pred_t = std::pair<size_t, float>;
    using pred_vec_t = std::vector<pred_t>;
    using path = std::experimental::filesystem::v1::path;

    struct classifier_t
    {
        classifier_t() {}

        cv::dnn::Net classifier;
        std::vector<cv::String> outlayers_names;

        enum CLASSES
        {
            ONE,
            TWO,
            SIZE
        };
        template <typename T> using clf_array = std::array<T, CLASSES::SIZE>;

        using parse_filename_func_t = bool(*)(const std::string & filename_short, clf_array<int> & gt_idx, clf_array<std::string> & gt_names);
        using change_filename_func_t = std::string(*)(const std::string & filename_short, const clf_array<std::string> & rec_names);

        struct param_t
        {
            /* loaded from cmd */
            clf_array<std::vector<std::string>> class_entries;
            std::string model;
            std::string weights;
            clf_array<path> filename;
            size_t image_size;
            bool check_filename;
            CLASSES num_classes;

            double scale_factor = 1.;
            cv::Scalar mean = cv::Scalar(0, 0, 0);
            bool swap_RB = false;
            bool inverse_channels = false;
            bool crop = false;
            int ddepth = 5;

            parse_filename_func_t filename_parser = nullptr;
            change_filename_func_t filename_changer = nullptr;
        } params;

        struct stat_t
        {
            stat_t() {};
            stat_t(size_t num_classes);

            /* loaded from cmd */
            size_t num_classes;
            size_t topk;
            float thresh;

            confusion_matrix conf;
            confusion_matrix conf_topk;

            void print(const param_t & params, const std::string & name = std::string(), const std::vector<std::string> & class_entries = std::vector<std::string>()) const;
            void add(size_t gt, const pred_vec_t & pred);
        };
        clf_array<stat_t> stat;

        struct clf_res_t
        {
            clf_array<std::string> gt;
            clf_array<std::string> rec;

            bool correct = false;
        };

        bool init_params(const cv::CommandLineParser & cmd);
        bool load_class_entries(const path & filename, const CLASSES id);
        bool load_classifier();
        bool parse_filename(const std::string & filename_short, clf_array<int> & gt_idx, clf_array<std::string> & gt_names) const;
        std::string change_filename(const std::string & filename_short, const clf_array<std::string> & rec_names) const;

        void print_stat() const;
        void process_file(const path & file, const cv::Mat & img, clf_res_t & result);
    };
}