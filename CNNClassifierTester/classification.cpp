#include "classification.h"

#include <logger.h>

#include <fstream>
#include <iomanip>

namespace clf
{
    using LL = logger::LOG_LEVEL_t;
    auto pair_desc = [](const std::pair<size_t, float> & l, const std::pair<size_t, float> & r) { return l.second > r.second; };

    bool classifier_t::init_params(const cv::CommandLineParser & cmd)
    {
        bool retval = true;

        if (!cmd.has("model"))
        {
            logger::LOG_MSG(LL::Error, "<model> must be specified.");
            retval = false;
        }
        params.model = cmd.get<std::string>("model");

        if (!cmd.has("weights"))
        {
            logger::LOG_MSG(LL::Error, "<weights> must be specified.");
            retval = false;
        }
        params.weights = cmd.get<std::string>("weights");
        params.check_filename = cmd.get<int>("filename_as_labels") == 0 ? false : true;
        params.image_size = cmd.get<size_t>("image_size");

        int classifier_mode = cmd.get<int>("classifier_mode");
        params.num_classes = (CLASSES)classifier_mode;

        if (classifier_mode > CLASSES::SIZE)
        {
            logger::LOG_MSG(LL::Error, "Wrong value <classifier_mode>=" + std::to_string(classifier_mode) + ". Allowed values: 1 - one class, 2 - two classes.");
            retval = false;
        }

        for (size_t i = 0; i < params.num_classes; ++i)
        {
            std::string id = std::to_string(i);

            if (!cmd.has("classes_" + id))
                continue;

            params.filename[i] = cmd.get<std::string>("classes_" + id);
            if (!load_class_entries(params.filename[i], (CLASSES)i))
            {
                logger::LOG_MSG(LL::Error, "Failed to load " + params.filename[i].string() + " file.");
                retval = false;
                continue;
            }

            if (params.class_entries[i].empty())
                continue;

            stat_t s(params.class_entries[i].size());
            s.topk= std::min(cmd.get<size_t>("topk_" + id), params.class_entries[i].size());
            s.thresh = cmd.has("classifier_threshold_" + id) ? cmd.get<float>("classifier_threshold_" + id) : 0.f;
            stat[i] = s;
        }

        return retval;
    }

    bool classifier_t::load_class_entries(const path & filename, const CLASSES id)
    {
        std::string filename_short = filename.filename().string().substr(0, filename.filename().string().find_last_of('.'));
        outlayers_names.push_back(filename_short);

        std::ifstream file(filename);
        if (!file.is_open())
            return false;

        std::string str;
        while (file >> str)
        {
            std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
            params.class_entries[id].push_back(str);
        }

        stat[id].conf = confusion_matrix(params.class_entries[id].size());
        stat[id].conf_topk = confusion_matrix(params.class_entries[id].size());
        return true;
    }

    bool classifier_t::load_classifier()
    {
        std::stringstream msg;
        msg << "\nClassifier: \n"
            << "Model: " << params.model << '\n'
            << "Config: " << params.weights << '\n'
            << "Num classes: " << params.num_classes;
        logger::LOG_MSG(LL::Info, msg.str());

        try
        {
            classifier = cv::dnn::readNet(params.model, params.weights);
            classifier.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            classifier.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

            if (classifier.empty())
            {
                logger::LOG_MSG(LL::Error, "Failed to load classifier.");
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

        logger::LOG_MSG(LL::Info, "Classifier is loaded.");

        return true;
    }

    bool classifier_t::parse_filename(const std::string & filename_short, clf_array<int> & gt_idx, clf_array<std::string> & gt_names) const
    {
        if (params.filename_parser)
            return params.filename_parser(filename_short, gt_idx, gt_names);

        /* default filename format:
        ** <class_1>[_<class_2>]_*.ext */
        size_t pos = 0;
        size_t of = 0;
        size_t substr = 0;

        size_t match = 0;
        while ((of = filename_short.find('_', pos)) != std::string::npos)
        {
           gt_names[substr] = filename_short.substr(pos, of - pos);

           bool have_match = false;
           for (size_t class_id = 0; !have_match && class_id < params.num_classes; ++class_id)
           {
               for (size_t i = 0; i < params.class_entries[class_id].size(); ++i)
                   if (params.class_entries[class_id][i] == gt_names[class_id])
                   {
                       gt_idx[class_id] = (int)i;
                       ++match;
                       have_match = true;
                       break;
                   }
           }

           pos = of + 1;

            //if (++substr >= params.num_classes)
            //    break;
            if (match == params.num_classes)
                break;
        }

        bool gt_ok = (match < params.num_classes ? false : true);
        for (size_t class_id = 0; gt_ok && class_id < params.num_classes; ++class_id)
            if (gt_idx[class_id] == -1)
                gt_ok = false;

        return gt_ok;
    }

    std::string classifier_t::change_filename(const std::string & filename_short, const clf_array<out_pred_vec_t> & rec_names) const
    {
        if (params.filename_changer)
            return params.filename_changer(filename_short, rec_names);

        /* filename format:
        ** <class_1>[_<class_2>]_*.ext */
        size_t of = 0;
        size_t count = 0;

        std::string new_filename;
        for (size_t class_id = 0; class_id < params.num_classes; ++class_id)
            //use top-1 recognition
            new_filename += rec_names[class_id][0].first + '_';

        if (filename_short.find('_') == std::string::npos)
            new_filename += filename_short;
        else
        {
            while ((of = filename_short.find('_', of)) != std::string::npos)
            {
                if (++count == params.num_classes)
                {
                    new_filename += filename_short.substr(of + 1);
                    break;
                }
                ++of;
            }
        }

        return new_filename;
    }

    void classifier_t::print_stat() const
    {
        for (size_t i = 0; i < params.num_classes; ++i)
        {
            stat[i].print(params, outlayers_names[i], params.class_entries[i]);
        }
    }

    void classifier_t::process_file(const path & file, const cv::Mat & img, clf_res_t & result)
    {
        cv::Mat crBlob = cv::dnn::blobFromImage(img,
            params.scale_factor,
            cv::Size((int)params.image_size, (int)params.image_size),
            params.mean,
            params.swap_RB,
            params.crop,
            params.ddepth);

        std::vector<cv::Mat> out;

        classifier.setInput(crBlob);
        classifier.forward(out, outlayers_names);

        clf_array<int> gt_idxes;
        gt_idxes.fill(-1);
        clf_array<std::string> gt_names;
        if (params.check_filename)
        {
            if (!parse_filename(file.filename().string(), gt_idxes, gt_names))
                return;
            result.gt = gt_names;
        }

        bool correct = true;
        for (size_t class_id = 0; class_id < params.num_classes; ++class_id)
        {
            pred_vec_t pred(params.class_entries[class_id].size());
            cv::Mat softmax_out = out[class_id].reshape(1, 1);

            for (size_t j = 0; j < softmax_out.cols; ++j)
                pred[j] = std::make_pair(j, softmax_out.at<float>(0, (int)j));

            std::sort(pred.begin(), pred.end(), pair_desc);

            for (size_t j = 0; j < stat[class_id].topk; ++j)
            {
                int idx = (int)pred[j].first;
                float rel = pred[j].second;

                if (rel < stat[class_id].thresh)
                    break;

                result.rec[class_id].push_back(std::make_pair(params.class_entries[class_id][idx], rel));

                if (j == 0)
                {
                    if (params.check_filename)
                    {
                        int gt_idx = gt_idxes[class_id];
                        if (gt_idxes[class_id] > 0)
                        {
                            correct &= ((size_t)gt_idx == idx);
                            stat[class_id].add((size_t)gt_idx, pred);
                        }
                        else
                            gt_idx = false;
                    }
                    else
                        stat[class_id].add(idx, pred);
                }
            }
        }

        result.correct = (params.check_filename && correct);
    }

    classifier_t::stat_t::stat_t(size_t num_classes)
        : num_classes(num_classes)
        , topk(0u)
        , conf(num_classes)
        , conf_topk(num_classes)
        , thresh(0.f)
    {
    }

    void classifier_t::stat_t::add(size_t gt, const pred_vec_t & pred)
    {
        size_t num_preds = pred.size();
        if (num_preds && pred[0].second >= thresh)
        {
            conf.add(gt, pred[0].first);

            if (topk > 1 && num_preds >= topk)
            {
                bool wrong = true;

                for (size_t i = 0; wrong && i < topk; ++i)
                    if (pred[i].first == gt)
                        wrong = false;

                wrong ? conf_topk.add(gt, pred[0].first) : conf_topk.add(gt, gt);
            }
        }
    }

    void classifier_t::stat_t::print(const param_t & params, const std::string & name, const std::vector<std::string> & class_entries) const
    {
        bool print_classnames = class_entries.size() == num_classes;
        bool print_conf = (num_classes < 30);

        if (print_conf)
        {
            std::stringstream msg;
            msg << '\n' << name << (name.empty() ? "Predictions:\n" : " predictions:\n");

            for (size_t class_id = 0; class_id < num_classes; ++class_id)
            {
                if (print_classnames)
                    msg << std::setw(15) << class_entries[class_id];
                for (size_t j = 0; j < num_classes; ++j)
                {
                    size_t pred = conf.get(class_id, j);
                    pred ?
                        (msg << std::setw(5) << pred << "(" << std::setw(3) << conf.getAccuracyPercent(class_id, j) << "%)") :
                        (msg << std::setw(5) << pred) << std::setw(6) << ' ';
                }
                msg << '\n';
            }

            if (topk > 1)
            {
                msg << '\n' << (name.empty() ? "TOP-": name + " TOP-") << topk << " predictions:\n";
                for (size_t class_id = 0; class_id < num_classes; ++class_id)
                {
                    if (print_classnames)
                        msg << std::setw(15) << class_entries[class_id];
                    for (size_t j = 0; j < num_classes; ++j)
                    {
                        size_t pred = conf_topk.get(class_id, j);
                        pred ?
                            (msg << std::setw(5) << pred << "(" << std::setw(3) << conf.getAccuracyPercent(class_id, j) << "%)") :
                            (msg << std::setw(5) << pred << std::setw(6) << ' ');
                    }
                    msg << '\n';
                }
            }
            logger::LOG_MSG(LL::Info, msg.str());
        }

        std::stringstream msg;
        msg << '\n' << (name.empty() ? "Accuracy:\n" : name + " accuracy:\n");
        msg << conf.getCorrect() << '/' << conf.size();
        if (conf.size())
            msg << '(' << std::setprecision(3) << 100. * conf.getCorrect() / conf.size() << "%)\n";

        if (topk > 1)
        {
            msg << '\n' << (name.empty() ? "TOP-" : name + " TOP-") << topk << " accuracy:\n";
            msg << conf_topk.getCorrect() << '/' << conf_topk.size();
            if (conf_topk.size())
                msg << '(' << std::setprecision(3) << 100. * conf_topk.getCorrect() / conf_topk.size() << "%)\n";
        }

    }
}