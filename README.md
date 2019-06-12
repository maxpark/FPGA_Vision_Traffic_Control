# System Design for Vision Based Traffic Sensing (FYP)
Customized digital architecture  to optimally implement a 20+ layer CNN on ZYNQ FPGA to detect traffic level from video feed and control traffic light timings accordingly.

![YOLOv2 inference on night time traffic data collected with our device at Piliyandala, Sri Lanka](docs/data_collection/4_yolo_output_2.png)

Group Members : Abarajithan G., Fonseka T.T., Wickramasinghe W.M.R.R., Wimalasuriya C. 
Supervisor    : Prof. Rohan Munasinghe 

# Abstract

We intend to design and demonstrate a highly optimized hardware architecture that can run a Convolutional Neural Network (CNN) or morphological operations in real time to deduce the level of traffic with acceptable accuracy while being robust to different lighting and weather from the video feed through a camera attached next to the traffic lights. The traffic level sensed through our system can be fed into algorithms developed by traffic engineers (who will be guiding us) to control the traffic lights and make the timing dynamic and sensitive to the traffic level for efficient traffic control. This is a part of the nationwide Intelligent Transport System (ITS) project done in collaboration with the Traffic Engineering Division, Department of Civil Engineering in our university and Road Development Authority (RDA) and funded by the World Bank. 

Currently available systems (in developed countries) for vehicle sensing include loop detectors, radar and wireless sensor networks. These methods fail to differentiate between heavy and light traffic conditions and occasionally produce false positives. Existing vision-based systems use outdated technology and they require expensive infrastructure unavailable in a developing country.  

Processing the video feed on edge using a CNN on a Field Programmable Gate Device (FPGA) for traffic sensing would be the low cost, scalable solution ideal for a developing country like Sri Lanka. However, it has not been attempted yet, since the compact neural networks and hardware that enable our approach were not available until recently. Hence, we conclude that our approach is unique and of great national importance. 

Potential benefits of our project include reduced traffic congestion, saving time and money of millions of people in developing countries. In addition to that, the hardware architecture we intend to develop as our major contribution in this project may be used for other similar applications such as in self driving cars with minimal modification. 

# Directory Structure

- /architecture     : Digital design and implementation
  - /design         : Design blueprints
  - /fpga           : Verilog files, Vivado projects
  - /simulations    : Modelsim simulations
- /collection_device: Rasberry pi / Nano codes and relevent device
- /data             : Datasets. Generally gitignored
  - /explore        : Jupyter notebooks for exploration
  - /preliminary    : Initial data, high_res are gitignored
- /docs             : Documentation, reports, figures, images of progress
- /nn               : Tensorflow / keras / numpy experiements on CNN architectures
  - /framework      : Keras-like numpy framework for testing CNNs
  - /yolov2         : Experiments and results on YOLOv2 network
- /other            : Permission letters and supporting docs
- /ref              : Research papers and videos. Large files gitignore'd
- /sim              : Simulator

## Rules

* Always gitignore large files and jupyter notebooks
* Start jupyter notebook names with your name: aba_yolov2_exp.ipynb
* Do not edit another person's jupyter notebook. Always copy and rename it with yours. Once experiment is complete, create a .py file with proper code.
