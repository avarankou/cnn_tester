#include "detection_utils.h"

#include <logger.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char ** argv)
{
    cv::CommandLineParser cmd(argc, argv,
        "{help h||help message}"
        /* detector params */
        "{model m||path to .xml file with model architecture}"
        "{weights w||path to .bin file with model weights}"
        "{image_size |300|classifier's input image size, 3 channels rgb image assumed}"
        "{threshold|0.5|detector threshold}"

        /* cropper params */
        "{indir||path to dir with test images}"
        "{indir_mode|-1|0 - root folder only, n - allowed subdirs depth (-1 for any depth)}"
        "{outdir||path to output dir with cropped images}"
        "{outdir_mode|0|-1 - disable output, 0 - common root folder, 1 - separate folders}"
        "{move_out|0|move images to output directory (copy by default)}"
        "{min_width|0.5|min object width relative to image width}"
        "{min_height|0.5|min object height relative to image width}"
        "{max_objects|1|max num of objects on single image}"
#       ifdef WITH_OPENCV_HIGHGUI
            "{debug_win dbg|0|output window with detection results}"
            "{recheck_falses recheck|0|manual recheck of objects of lower size and images without detected images}"
#       endif // WITH_OPENCV_HIGHGUI
    );

    cmd.about("Program runs traffic detector (defined by <model> & <weights>) over <indir> and copies result to <outdir> (move_out = 1 to move, outdir_mode = -1 to disable output)."
#       ifdef WITH_OPENCV_HIGHGUI
            "\nIf <debug_win> is set to 1 each image is shown before saving."
            "\nIf <recheck_falses> is set to 1 objects of lower size and images without detections are shown before skipping."
#       endif // WITH_OPENCV_HIGHGUI
    );

    try
    {
        cmd.printMessage();

        if (!dt::init_params(cmd))
            return EXIT_FAILURE;

        dt::process_dir();
    }
    catch (std::exception & e)
    {
        logger::LOG_MSG(logger::LOG_LEVEL_t::Error, std::string("Exception in main():\n") + e.what());
    }
    catch (...)
    {
        logger::LOG_MSG(logger::LOG_LEVEL_t::Error, "Unknown exception in main()");
    }

    return EXIT_SUCCESS;
}