#include "classification_utils.h"

#include <logger.h>

int main(int argc, char ** argv)
{
    cv::CommandLineParser cmd(argc, argv,
        "{help h||help message}"
        /* classifier params */
        "{model m||path to .xml file with model architecture}"
        "{weights w||path to .bin file with model weights}"
        "{filename_as_labels|0|filename in format \"<class_1>[_<class_2>]_*.ext\" if true gt labels from filename compared to classification results}"
        "{image_size |72|classifier's input image size, 3 channels rgb image assumed}"
        "{classifier_mode c|1|1 - single-class classification, 2 - two-classes classification}"
        "{classifier_threshold_0|0.3|first classifier threshold}"
        "{classes_0|first.txt|path to .txt file with listed classes for first classifier (must be equal to net's output node's name)}"
        "{topk_0 |1|topk predictions in statistics for first classifier (alongside with topk = 1)}"
        "{classifier_threshold_1|0.3|second classifier threshold}"
        "{classes_1|second.txt|path to .txt file with listed classes for second classifier (must be equal to net's output node's name)}"
        "{topk_1 |1|topk predictions in statistics for second classifier (alongside with topk = 1)}"

        /* classifier tester params */
        "{indir||path to dir with test images}"
        "{indir_mode|0|0 - root folder only, 1 - allowed subdirs depth (-1 for any depth)}"
        "{outdir||path to output dir for classification results}"
        "{outdir_mode|0|-1 - disable output, 0 - common root folder, 1 - separate folders}"
        "{out_filename|0|0 - with filename from classified labels, 1 - with original filename}"
        "{move_out|0|move images to output directory (copy by default)}"
        "{annotation|0|classification annotation of output images}"
        "{save_misclassified mis|0|save misclassified images (only if filename_as_labels = 1): 0 - with filename from classified labels, 1 - with original filename, -1 - don't save misclasified}"
        "{misclassified_dir misdir||path to output dir for misclassified images}"
#       ifdef WITH_OPENCV_HIGHGUI
            "{debug_win dbg|0|output window with image classification results}"
            "{recheck_misclassified recheck|0|manual recheck of misclassified images}"
#       endif // WITH_OPENCV_HIGHGUI
        
    );

    cmd.about("Program runs classifier (defined by <model> & <weights>) over <indir> and copies result to <outdir> (move_out = 1 to move)."
        "\nDepending on <classifier_mode> one output class (classifier_mode = 1), or two output classes (classifier_mode = 2) are tested."
        "\nClass outputs must be listed in <classes_1> and <classes_2> text files (filenames must be equal to net's output layers' names)."
        "\nInput images from <indir> are saved to <outdir>.\n"
        "\n<outdir_mode>=-1 to disable output, 0 for single out folder, 1 for folder hierarchy simmilar to one in <indir>.\n"
        "\n<out_filename>=0 to save file in <outdir> with filename from classified labels, 1 to save with original filename.\n"
        "\nMisclassified images are saved to <misdir> (<save_misclassified> = -1 to disable) with original filename (= 0) of with new filename based on classification (= 1).\n"
        "\nIf <save_misclassified> != 0 input image filename format: \"<first_class>[_<second_class>]_*.<jpeg | bmp | gif>\".\n"
#       ifdef WITH_OPENCV_HIGHGUI
            "\nIf <debug_win> is set to 1 each image is shown before saving."
            "\nIf <recheck_misclassified> is set to 1 misclassified images are shown before saving (with filename depended on <save_misclassified>)."
#       endif // WITH_OPENCV_HIGHGUI
    );

    try
    {
        cmd.printMessage();
    
        if (!ct::init_params(cmd))
            return EXIT_FAILURE;
    
        ct::process_dir();
    }
    catch (std::exception & e)
    {
        logger::LOG_MSG(logger::LOG_LEVEL_t::Error, std::string("Exception in main():\n" )+ e.what());
    }
    catch (...)
    {
        logger::LOG_MSG(logger::LOG_LEVEL_t::Error, "Unknown exception in main()");
    }

    return EXIT_SUCCESS;
}