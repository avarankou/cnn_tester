CNNTester

Small OpenCV-based utility to test trained CNN on existing image dataset.
Currently includes two projects: CNNDetectorTester and CNNClassifierTester.

CNNDetectorTester allows to see how pretrained detector performs on input images. Also it may be used to crop detections from images (with all adjustable params: detection reliability, suitable object size, max number of objects etc.) with optional manual recheck.

CNNClassifierTester alloows to see how pretrained classifier performs on input images and prediction confusion matrix (top1 and topN). It allows single-class or two-class classification (more classes may be added easily). Also it may be used to label existing images with optional manual recheck.

Both projects provide variety of reasonable command line arguments to set CNN params and dataset input/output pathes for ease of use in scripts.
Both projects use FiletreeRambler utility for easy and efficient filesystem crawling.