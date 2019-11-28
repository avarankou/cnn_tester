#include "detection.h"

#include <logger.h>

namespace detector
{
    using LL = logger::LOG_LEVEL_t;
    auto td_res_desc = [](const detector_t::det_res_t & l, const detector_t::det_res_t & r) { return l.prob > r.prob; };

    bool detector_t::init_params(const cv::CommandLineParser & cmd)
    {
        bool retval = true;

        if (!cmd.has("model"))
        {
            logger::LOG_MSG(LL::Error, "<model> must be specified.");
            retval = false;
        }
        params.model = path(cmd.get<std::string>("model"));

        if (!cmd.has("weights"))
        {
            logger::LOG_MSG(LL::Error, "<weights> must be specified.");
            retval = false;
        }
        params.weights = path(cmd.get<std::string>("weights"));

        params.image_size = cmd.get<size_t>("image_size");

        params.thresh = cmd.has("threshold") ? cmd.get<float>("threshold") : 0.f;

        return retval;
    }


    bool detector_t::load_detector()
    {
        std::stringstream msg;

        msg << "\nTraffic detector: \n";
        msg
            << "Model: " << params.model << '\n'
            << "Config: " << params.weights;

        logger::LOG_MSG(LL::Info, msg.str());

        try
        {
            net = cv::dnn::readNet(params.model.string(), params.weights.string());
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

            if (net.empty())
            {
                logger::LOG_MSG(LL::Error, "Failed to load detector.");
                return false;
            }
        }
        catch (cv::Exception & e)
        {
            logger::LOG_MSG(LL::Error, e.what());
            return false;
        }
        catch (std::exception & e)
        {
            logger::LOG_MSG(LL::Error, e.what());
            return false;
        }

        logger::LOG_MSG(LL::Info, "Detector is loaded.");
        return true;
    }

    void detector_t::process_file(const cv::Mat & img, std::vector<det_res_t> & results)
    {
        // detector
        cv::Mat detBlob = cv::dnn::blobFromImage(img,
            params.scale_factor,
            cv::Size((int)params.image_size, (int)params.image_size),
            params.mean,
            params.inverse_channels,
            params.crop,
            params.ddepth);

        net.setInput(detBlob);

        cv::Mat output = net.forward();

        cv::Mat reshaped{ output.size[2], output.size[3], CV_32F, output.ptr<float>() };

        for (int i = 0; i < (int)reshaped.rows; ++i)
        {
            float score = reshaped.at<float>(i, 2);
            if (score > params.thresh)
            {
                det_res_t result;
                result.class_id = (unsigned int)reshaped.at<float>(i, 1);
                result.x = std::max(0, int(reshaped.at<float>(i, 3) * img.cols));
                result.y = std::max(0, int(reshaped.at<float>(i, 4) * img.rows));
                result.w = std::max(0, int(reshaped.at<float>(i, 5) * img.cols - result.x));
                result.h = std::max(0, int(reshaped.at<float>(i, 6) * img.rows - result.y));
                result.prob = score;

                results.push_back(result);
            }
        }

        std::sort(results.begin(), results.end(), td_res_desc);
    }
}