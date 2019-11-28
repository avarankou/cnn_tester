#pragma once

#include <opencv2/dnn.hpp>

#include <filesystem>

namespace detector
{
    using path = std::experimental::filesystem::v1::path;

    struct detector_t
    {
        detector_t() {}

        cv::dnn::Net net;

        struct param_t
        {
            /* loaded from cmd */
            path model;
            path weights;
            size_t image_size;
            float thresh;

            double scale_factor = 1.;
            cv::Scalar mean = cv::Scalar(0, 0, 0);
            bool swap_RB = false;
            bool inverse_channels = false;
            bool crop = false;
            int ddepth = 5;
        } params;

        struct det_res_t
        {
            int class_id;
            int x, y;
            int w, h;
            float prob;
        };

        bool init_params(const cv::CommandLineParser & cmd);
        bool load_detector();

        void process_file(const cv::Mat & img, std::vector<det_res_t> & results);
    };
}