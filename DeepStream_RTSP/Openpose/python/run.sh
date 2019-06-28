#!/bin/bash

echo "Performing model calibration... Please wait, this may take some time..."
python3 to_int8.py --dataset ../../ --model_file ../../model/graph_opt.uff --save_name ../../model/openpose_int8.trt

python3 generate_config.py
deepstream-app -c ../openpose_config.txt > /dev/null