/**
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include "nvdssample_Openpose_common.h"
#include "find_peaks.h"
#include <fstream>
#include <iomanip>

#define CLIP(a,min,max) (MAX(MIN(a, max), min))
/* This is a sample bounding box parsing function for the sample FasterRCNN
 * detector model provided with the TensorRT samples. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomOpenPose (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferParseObjectInfo> &objectList)
{
  static int PredLayerIndex = -1;

  if (PredLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "Openpose/concat_stage7") == 0) {
        PredLayerIndex = i;
        break;
      }
    }
    if (PredLayerIndex == -1) {
    std::cerr << "Could not find Openpose/concat_stage7 layer buffer while parsing" << std::endl;
    return false;
    }
  }


  auto* buffer = (float *) outputLayersInfo[PredLayerIndex].buffer;
  vector<Peak> peaks = OpenposePostProc(buffer, batch_size);

  for(const auto& peak: peaks){
      NvDsInferParseObjectInfo object;
      object.left =peak.x -5;
      object.top = peak.y -5;
      object.height = 10;
      object.width = 10;
      object.classId = 15;

      object.detectionConfidence = 0.8;
      objectList.push_back(object);
  }

  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomOpenPose);
