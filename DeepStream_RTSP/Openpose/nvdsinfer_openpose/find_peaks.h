//
// Created by akorovko on 6/25/19.
//

#ifndef NVDSINFER_OPENPOSE_FIND_PEAKS_H
#define NVDSINFER_OPENPOSE_FIND_PEAKS_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "nvdssample_Openpose_common.h"

using namespace std;
using namespace cv;

struct Peak{
    int x;
    int y;
    Peak(const int& x, const int& y) : x(x), y(y){}
};


vector<Mat> get_heatmaps(float* buffer, int size){
    vector<Mat> out;

    int batches_num = size/batch_size;
    assert(batches_num == 1);

    float* temp = buffer;
    Mat image(batch_width, batch_height, CV_32FC(batch_depth));
    memcpy(image.data, temp, batch_size * sizeof(float));

    vector<Mat> mat_arr;
    split(image, mat_arr);
//    for(int j = 0; j < heatmap_depth; j++){
//        out.push_back(mat_arr[j]);
//    }
    out.push_back(mat_arr[1]);
    return out;
}

vector<Mat> resize_heatmaps(const vector<Mat>& heatmaps){
    vector<Mat> out;
    for(const auto& heatmap: heatmaps){
        Mat B;
        resize(heatmap, B, Size(image_width, image_height), INTER_CUBIC);
        out.push_back(B);
    }
    return out;
}


vector<Peak> process_heatmap(Mat& heatmap, float thresh){
    GaussianBlur(heatmap, heatmap, Size(0, 0), 3, 3);
    Size s = heatmap.size();
    vector<Peak> out;
    for(int i = 1; i < s.height -1; i++){
        for(int j = 1; j < s.width -1; j++){
            if(
                    (heatmap.at<float>(i,j) > heatmap.at<float>(i+1,j)) &&
                    (heatmap.at<float>(i,j) > heatmap.at<float>(i,j+1)) &&
                    (heatmap.at<float>(i,j) > heatmap.at<float>(i-1,j)) &&
                    (heatmap.at<float>(i,j) > heatmap.at<float>(i,j-1)) &&
                    (heatmap.at<float>(i,j) > thresh)
                    ){
                out.push_back({j,i});
            }
        }
    }
    return out;
}

vector<Peak> OpenposePostProc(float* buffer, int size){
    auto heatmaps = get_heatmaps(buffer, size);
    auto n_heatmaps = resize_heatmaps(heatmaps);


    vector<Peak> out;
    for(auto& hm: n_heatmaps){
        auto vect = process_heatmap(hm, thresh);
        out.insert(end(out), begin(vect), end(vect));
    }

//    Mat zero = Mat::zeros(368, 656, CV_8UC1);
//
//    for(auto& peak: out){
//        circle(zero, Point(peak.x, peak.y), 2, Scalar(255, 1));
//    }
//    imwrite("/home/akorovko/Code/points.png", zero);
//    exit(-1);


    return out;
}
#endif //NVDSINFER_OPENPOSE_FIND_PEAKS_H
