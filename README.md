# Realtime-Object-Detection (with webcam on macOS and Raspberry Pi)

This is a very small Python raltime object detection implementation tested on macOS (12 Monterey) using the build in webcam and OpenCV. NOTE: Please load the pretrained Deep Learning weights file from here --> https://pjreddie.com/darknet/yolo/ (yolov3.weights 237MB) and place it in the model folder. The file is to large (exceeds 100MB) for github and must be loaded an copied by hand. It should run on amy OS.

![alt text](demo.png)

Start with: `python RealTimeObjectDetection.py`
The RTOD uses Darknet, Yolo and OpenCV. The only Library you have to be installed on your computer is opencv-python in `pip`or `conda`. You should trim the camera gamma factor. 
